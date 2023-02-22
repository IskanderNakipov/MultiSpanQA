QUESTIONS = {
    'rank': "Какое звание у {}?",
    'dolls': "Какая должность у {}?",
    'medals': "Какие награды получил {}?",
    'age': "Сколько лет было {}?",
    'death_date': "Когда погиб {}?",
    'death_place': "Где погиб {}?",
    'birth_date': "Когда родился  {}?",
    'birth_place': "Где родился  {}?",
    'burial_date': "Когда было прощание с {}?",
    'burial_place': "Где было прощание с  {}?",
}

STRUCTURE_LIST = [
    'rank',
    'dolls',
    'medals',
    'age',
    'death_date',
    'death_place',
    'birth_date',
    'birth_place',
    'burial_date',
    'burial_place'
]

STRUCTURE_TO_ID = {l: i for i, l in enumerate(STRUCTURE_LIST)}