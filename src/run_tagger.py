import os
import logging
import collections
from tqdm.auto import tqdm
from typing import Optional, Tuple

import torch
import numpy as np
from datasets import load_dataset

from transformers import (
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.utils.versions import require_version

from arguments import ModelArguments, DataTrainingArguments
from common_utils import delete_last_checkpoint, base_init, train_dataset_preprocessing
from constants import STRUCTURE_TO_ID, STRUCTURE_LIST
from logging_utils import setup_logging, logger
from models import BertTaggerForMultiSpanQA, RobertaTaggerForMultiSpanQA
from trainer import QuestionAnsweringTrainer
from eval_script import *

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")
logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"


def postprocess_tagger_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    id2label,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
    save_embeds = False,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """

    # print(len(predictions),predictions)
    if len(predictions[0].shape) != 1: # Not CRF output
        if predictions[0].shape[-1] != 3:
            raise RuntimeError(f"`predictions` should be in shape of (max_seq_length, 3).")
        all_logits = predictions[0]
        all_hidden = predictions[1]
        all_labels = np.argmax(predictions[0], axis=2)

        if len(predictions[0]) != len(features):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")
    else:
        all_logits = predictions

    if -100 not in id2label.values():
        id2label[-100] = 'O'

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_ids = []
    all_valid_logits = []
    all_valid_hidden = []

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            sequence_ids = features[feature_index]['sequence_ids']
            word_ids = features[feature_index]['word_ids']
            logits = [l for l in all_logits[feature_index]]
            hidden = [l for l in all_hidden[feature_index]]
            labels = [id2label[l] for l in all_labels[feature_index]]
            prelim_predictions.append(
                {
                    "logits": logits,
                    "hidden": hidden,
                    "labels": labels,
                    "word_ids": word_ids,
                    "sequence_ids": sequence_ids
                }
            )

        previous_word_idx = -1
        ignored_index = []  # Some example tokens will disappear after tokenization.
        valid_labels = []
        valid_logits = []
        valid_hidden = []
        for x in prelim_predictions:
            logits = x['logits']
            hidden = x['hidden']
            labels = x['labels']
            word_ids = x['word_ids']
            sequence_ids = x['sequence_ids']

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            for word_idx, label, lo, hi in list(zip(word_ids,labels,logits,hidden))[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    continue
                # We set the label for the first token of each word.
                elif word_idx > previous_word_idx:
                    ignored_index += range(previous_word_idx+1, word_idx)
                    valid_labels.append(label)
                    valid_logits.append(lo)
                    valid_hidden.append(hi)
                    previous_word_idx = word_idx
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    continue

        context = example["context"]
        for i in ignored_index[::-1]:
            context = context[:i] + context[i+1:]
        assert len(context) == len(valid_labels)

        predict_entities = get_entities(valid_labels, context)
        predictions = [x[0] for x in predict_entities]
        all_predictions[example["id"]] = predictions

        all_ids.append(example["id"])
        all_valid_logits.append(valid_logits)
        all_valid_hidden.append(valid_hidden)

    all_valid_logits = np.array(all_valid_logits)
    all_valid_hidden = np.array(all_valid_hidden)

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if save_embeds:
        logger.info(f"Saving embeds for CRF.")
        ids_file = os.path.join(output_dir, "ids.json" if prefix is None else f"{prefix}_ids.json")
        with open(ids_file, "w") as writer:
            writer.write(json.dumps(all_ids, indent=4) + "\n")

        logits_file = os.path.join(output_dir, "logits.np" if prefix is None else f"{prefix}_logits.np")
        hidden_file = os.path.join(output_dir, "hidden.np" if prefix is None else f"{prefix}_hidden.np")
        with open(logits_file, "wb") as f1:
            np.save(f1, all_valid_logits)
        with open(hidden_file, "wb") as f1:
            np.save(f1, all_valid_hidden)

    return prediction_file


def main():
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = setup_logging(training_args)

    # Detecting last checkpoint.
    last_checkpoint = delete_last_checkpoint(training_args, logger)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {'train': os.path.join(data_args.data_dir, data_args.train_file),
                  'validation': os.path.join(data_args.data_dir, "valid.json")}
    if training_args.do_predict:
        data_files['test'] = os.path.join(data_args.data_dir, "test.json")

    raw_datasets = load_dataset('json', field='data', data_files=data_files)

    question_column_name = data_args.question_column_name
    context_column_name = data_args.context_column_name
    label_column_name = data_args.label_column_name

    structure_list = STRUCTURE_LIST
    structure_to_id = STRUCTURE_TO_ID

    label_list = ["B", "I", "O"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    config, tokenizer = base_init(data_args, id2label, label2id, model_args, num_labels)

    if 'roberta' in model_args.model_name_or_path:
        model = RobertaTaggerForMultiSpanQA.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = BertTaggerForMultiSpanQA.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        plus = False
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name],
            examples[context_column_name],
            truncation="only_second",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=padding,
            is_split_into_words=True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["labels"] = []
        if plus:
            tokenized_examples["num_span"] = []
        tokenized_examples["structure"] = []
        tokenized_examples["example_id"] = []
        tokenized_examples["word_ids"] = []
        tokenized_examples["sequence_ids"] = []

        for i, sample_index in enumerate(sample_mapping):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            label = examples[label_column_name][sample_index]
            word_ids = tokenized_examples.word_ids(i)
            previous_word_idx = None
            label_ids = [-100] * token_start_index

            for word_idx in word_ids[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            tokenized_examples["labels"].append(label_ids)
            if plus:
                tokenized_examples["num_span"].append(float(label_ids.count(0))) # count num of B as num_spans
            # tokenized_examples["num_span"].append(examples['num_span'][sample_index] / data_args.max_num_span)

            tokenized_examples["structure"].append(
                structure_to_id[examples['structure'][sample_index] if 'structure' in examples else '']
            )
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["word_ids"].append(word_ids)
            tokenized_examples["sequence_ids"].append(sequence_ids)
        return tokenized_examples

    # Preprocessing is slightly different for training and evaluation.
    train_column_names = raw_datasets["train"].column_names
    column_names = train_column_names
    if training_args.do_train or data_args.save_embeds:
        train_dataset = train_dataset_preprocessing(column_names, data_args, prepare_train_features, raw_datasets,
                                                    training_args)

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name],
            examples[context_column_name],
            truncation="only_second",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=padding,
            is_split_into_words=True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        tokenized_examples["word_ids"] = []
        tokenized_examples["sequence_ids"] = []

        for i, sample_index in enumerate(sample_mapping):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            word_ids = tokenized_examples.word_ids(i)
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["word_ids"].append(word_ids)
            tokenized_examples["sequence_ids"].append(sequence_ids)

        return tokenized_examples

    if training_args.do_eval:
        eval_examples = raw_datasets["validation"]
        eval_column_names = raw_datasets["validation"].column_names
        # Validation Feature Creation
        if data_args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        predict_examples = raw_datasets["test"]
        test_column_names = raw_datasets["test"].column_names
        # Predict Feature Creation
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=test_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        pred_file = postprocess_tagger_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            id2label=id2label,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
            save_embeds=data_args.save_embeds,
        )
        return pred_file

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        data_files=data_files,  # for quick evaluation
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=multi_span_evaluate_from_file,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics(split="train", metrics=metrics)
        trainer.save_metrics(split="train", metrics=metrics)
        trainer.save_state()
    else:
        model.load_state_dict(torch.load(os.path.join(training_args.output_dir,'pytorch_model.bin')))
        trainer.model = model

    if data_args.save_embeds:
        logger.info("*** Evaluate on Train ***")
        metrics = trainer.evaluate(eval_dataset=train_dataset, eval_examples=train_examples,  metric_key_prefix="train")
        metrics["train_samples"] = len(train_examples)
        trainer.log_metrics(split="train", metrics=metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_examples)
        trainer.log_metrics(split="eval", metrics=metrics)
        trainer.save_metrics(split="eval", metrics=metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        metrics = trainer.predict(predict_dataset, predict_examples)
        metrics["predict_samples"] = len(predict_examples)
        trainer.log_metrics(split="predict", metrics=metrics)
        trainer.save_metrics(split="predict", metrics=metrics)


if __name__ == "__main__":
    main()
