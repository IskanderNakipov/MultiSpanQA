import collections
import os

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    DataCollatorForTokenClassification,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)
from transformers.utils.versions import require_version

from arguments import ModelArgumentsPlus, DataTrainingArgumentsPlus
from common_utils import delete_last_checkpoint, base_init, train_dataset_preprocessing
from constants import STRUCTURE_LIST, STRUCTURE_TO_ID
from eval_script import *
from logging_utils import setup_logging, logger
from models import TaggerPlusForMultiSpanQA

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

os.environ["WANDB_DISABLED"] = "true"


def main():
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArgumentsPlus, DataTrainingArgumentsPlus, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setup_logging(training_args)

    # Detecting last checkpoint.
    delete_last_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {'train': os.path.join(data_args.data_dir, data_args.train_file),
                  'validation':os.path.join(data_args.data_dir, "valid.json")}
    if training_args.do_predict:
                  data_files['test'] = os.path.join(data_args.data_dir, "test.json")
    raw_datasets = load_dataset('json', field='data', data_files=data_files)

    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features

    question_column_name = data_args.question_column_name
    context_column_name = data_args.context_column_name
    label_column_name = data_args.label_column_name

    structure_list = STRUCTURE_LIST
    structure_to_id = STRUCTURE_TO_ID

    label_list = ["B", "I", "O"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        b_to_i_label.append(label_list.index(label.replace("B", "I")))

    config, tokenizer = base_init(data_args, id2label, label2id, model_args, num_labels)

    model = TaggerPlusForMultiSpanQA.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        structure_lambda=model_args.structure_lambda,
        span_lambda=model_args.span_lambda,
    )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        plus = True
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

    if training_args.do_train or data_args.save_embeds:
        train_dataset = train_dataset_preprocessing(column_names, data_args, prepare_train_features, raw_datasets,
                                                    training_args)

    if training_args.do_eval:
        eval_examples = raw_datasets["validation"]
        # Validation Feature Creation
        if data_args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        predict_examples = raw_datasets["test"]
        # Predict Feature Creation
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    tmp_train_dataset = train_dataset.remove_columns(["example_id","word_ids","sequence_ids"])
    tmp_eval_dataset = eval_dataset.remove_columns(["example_id","word_ids","sequence_ids"])

    # Run without Trainer

    import math

    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from transformers import (
        AdamW,
        get_scheduler,
    )

    accelerator = Accelerator()
    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    )

    train_dataloader = DataLoader(
        tmp_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        tmp_eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs[0]
            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

    # evaluate
    model.eval()
    all_p = []
    all_span_p = []
    all_struct_p = []
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            _, p, span_p, _, _ = model(**batch)
            all_p.append(p.cpu().numpy())
            all_span_p.append(span_p.cpu().numpy())

    all_p = [i for x in all_p for i in x]
    all_span_p = np.concatenate(all_span_p)

    # Post processing
    features = eval_dataset
    examples = eval_examples
    if len(all_p) != len(features):
        raise ValueError(f"Got {len(all_p[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_confs = collections.OrderedDict()
    all_nums = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

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
            confs = [np.max(l) for l in all_p[feature_index]]
            logits = [np.argmax(l) for l in all_p[feature_index]]
            labels = [id2label[l] for l in logits]
            nums = all_span_p[feature_index]
            prelim_predictions.append(
                {
                    "nums": nums,
                    "confs": confs,
                    "logits": logits,
                    "labels": labels,
                    "word_ids": word_ids,
                    "sequence_ids": sequence_ids
                }
            )

        previous_word_idx = -1
        ignored_index = [] # Some example tokens will be disappear after tokenization.
        valid_labels = []
        valid_confs = []
        valid_nums = sum(list(map(lambda x: x['nums'], prelim_predictions)))
        for x in prelim_predictions:
            confs = x['confs']
            labels = x['labels']
            word_ids = x['word_ids']
            sequence_ids = x['sequence_ids']

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            for word_idx,label,conf in list(zip(word_ids,labels,confs))[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    continue
                # We set the label for the first token of each word.
                elif word_idx > previous_word_idx:
                    ignored_index += range(previous_word_idx+1,word_idx)
                    valid_labels.append(label)
                    valid_confs.append(str(conf))
                    previous_word_idx = word_idx
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    continue

        context = example["context"]
        for i in ignored_index[::-1]:
            context = context[:i] + context[i+1:]
        assert len(context) == len(valid_labels) == len(valid_confs)

        predict_entities = get_entities(valid_labels, context)
        predict_confs = get_entities(valid_labels, valid_confs)
        confidence = [x[0] for x in predict_confs]
        predictions = [x[0] for x in predict_entities]
        all_predictions[example["id"]] = predictions
        all_confs[example['id']] = confidence
        all_nums[example["id"]] = valid_nums

    # Evaluate on valid
    golds = read_gold(os.path.join(data_args.data_dir, "valid.json"))
    print(multi_span_evaluate(all_predictions, golds))
    # Span adjustment
    for key in all_predictions.keys():
        if len(all_predictions[key]) > math.ceil(all_nums[key]*21):
            confs = list(map(lambda x: max([float(y) for y in x.split()]), all_confs[key]))
            new_preds = sorted(zip(all_predictions[key],confs), key=lambda x: x[1], reverse=True)[:math.ceil(all_nums[key]*21)]
            new_preds = [x[0] for x in new_preds]
            all_predictions[key] = new_preds
    # Evaluate again
    print(multi_span_evaluate(all_predictions, golds))


if __name__ == "__main__":
    main()
