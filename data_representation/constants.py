import numpy as np

# for chord analysis
NUM2PITCH = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
}

# referred to mmt "https://github.com/salu133445/mmt"
PROGRAM_INSTRUMENT_MAP = {
    # Pianos
    0: "piano",
    1: "piano",
    2: "piano",
    3: "piano",
    4: "electric-piano",
    5: "electric-piano",
    6: "harpsichord",
    7: "clavinet",
    # Chromatic Percussion
    8: "celesta",
    9: "glockenspiel",
    10: "music-box",
    11: "vibraphone",
    12: "marimba",
    13: "xylophone",
    14: "tubular-bells",
    15: "dulcimer",
    # Organs
    16: "organ",
    17: "organ",
    18: "organ",
    19: "church-organ",
    20: "organ",
    21: "accordion",
    22: "harmonica",
    23: "bandoneon",
    # Guitars
    24: "nylon-string-guitar",
    25: "steel-string-guitar",
    26: "electric-guitar",
    27: "electric-guitar",
    28: "electric-guitar",
    29: "electric-guitar",
    30: "electric-guitar",
    31: "electric-guitar",
    # Basses
    32: "bass",
    33: "electric-bass",
    34: "electric-bass",
    35: "electric-bass",
    36: "slap-bass",
    37: "slap-bass",
    38: "synth-bass",
    39: "synth-bass",
    # Strings
    40: "violin",
    41: "viola",
    42: "cello",
    43: "contrabass",
    44: "strings",
    45: "strings",
    46: "harp",
    47: "timpani",
    # Ensemble
    48: "strings",
    49: "strings",
    50: "synth-strings",
    51: "synth-strings",
    52: "voices",
    53: "voices",
    54: "voices",
    55: "orchestra-hit",
    # Brass
    56: "trumpet",
    57: "trombone",
    58: "tuba",
    59: "trumpet",
    60: "horn",
    61: "brasses",
    62: "synth-brasses",
    63: "synth-brasses",
    # Reed
    64: "soprano-saxophone",
    65: "alto-saxophone",
    66: "tenor-saxophone",
    67: "baritone-saxophone",
    68: "oboe",
    69: "english-horn",
    70: "bassoon",
    71: "clarinet",
    # Pipe
    72: "piccolo",
    73: "flute",
    74: "recorder",
    75: "pan-flute",
    76: None,
    77: None,
    78: None,
    79: "ocarina",
    # Synth Lead
    80: "lead",
    81: "lead",
    82: "lead",
    83: "lead",
    84: "lead",
    85: "lead",
    86: "lead",
    87: "lead",
    # Synth Pad
    88: "pad",
    89: "pad",
    90: "pad",
    91: "pad",
    92: "pad",
    93: "pad",
    94: "pad",
    95: "pad",
    # Synth Effects
    96: None,
    97: None,
    98: None,
    99: None,
    100: None,
    101: None,
    102: None,
    103: None,
    # Ethnic
    104: "sitar",
    105: "banjo",
    106: "shamisen",
    107: "koto",
    108: "kalimba",
    109: "bag-pipe",
    110: "violin",
    111: "shehnai",
    # Percussive
    112: None,
    113: None,
    114: "steel-drums",
    115: None,
    116: None,
    117: "melodic-tom",
    118: "synth-drums",
    119: "synth-drums",
    # Sound effects
    120: None,
    121: None,
    122: None,
    123: None,
    124: None,
    125: None,
    126: None,
    127: None,
}

# referred to mmt "https://github.com/salu133445/mmt"
INSTRUMENT_PROGRAM_MAP = {
    # Pianos
    "piano": 0,
    "electric-piano": 4,
    "harpsichord": 6,
    "clavinet": 7,
    # Chromatic Percussion
    "celesta": 8,
    "glockenspiel": 9,
    "music-box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular-bells": 14,
    "dulcimer": 15,
    # Organs
    "organ": 16,
    "church-organ": 19,
    "accordion": 21,
    "harmonica": 22,
    "bandoneon": 23,
    # Guitars
    "nylon-string-guitar": 24,
    "steel-string-guitar": 25,
    "electric-guitar": 26,
    # Basses
    "bass": 32,
    "electric-bass": 33,
    "slap-bass": 36,
    "synth-bass": 38,
    # Strings
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "harp": 46,
    "timpani": 47,
    # Ensemble
    "strings": 49,
    "synth-strings": 50,
    "voices": 52,
    "orchestra-hit": 55,
    # Brass
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "horn": 60,
    "brasses": 61,
    "synth-brasses": 62,
    # Reed
    "soprano-saxophone": 64,
    "alto-saxophone": 65,
    "tenor-saxophone": 66,
    "baritone-saxophone": 67,
    "oboe": 68,
    "english-horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    # Pipe
    "piccolo": 72,
    "flute": 73,
    "recorder": 74,
    "pan-flute": 75,
    "ocarina": 79,
    # Synth Lead
    "lead": 80,
    # Synth Pad
    "pad": 88,
    # Ethnic
    "sitar": 104,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "kalimba": 108,
    "bag-pipe": 109,
    "shehnai": 111,
    # Percussive
    "steel-drums": 114,
    "melodic-tom": 117,
    "synth-drums": 118,
}

FINED_PROGRAM_INSTRUMENT_MAP ={
    # Pianos
    0: "Acoustic-Grand-Piano",
    1: "Bright-Acoustic-Piano",
    2: "Electric-Grand-Piano",
    3: "Honky-Tonk-Piano",
    4: "Electric-Piano-1",
    5: "Electric-Piano-2",
    6: "Harpsichord",
    7: "Clavinet",

    # Chromatic Percussion
    8: "Celesta",
    9: "Glockenspiel",
    10: "Music-Box",
    11: "Vibraphone",
    12: "Marimba",
    13: "Xylophone",
    14: "Tubular-Bells",
    15: "Dulcimer",

    # Organs
    16: "Drawbar-Organ",
    17: "Percussive-Organ",
    18: "Rock-Organ",
    19: "Church-Organ",
    20: "Reed-Organ",
    21: "Accordion",
    22: "Harmonica",
    23: "Tango-Accordion",

    # Guitars
    24: "Acoustic-Guitar-nylon",
    25: "Acoustic-Guitar-steel",
    26: "Electric-Guitar-jazz",
    27: "Electric-Guitar-clean",
    28: "Electric-Guitar-muted",
    29: "Overdriven-Guitar",
    30: "Distortion-Guitar",
    31: "Guitar-harmonics",

    # Basses
    32: "Acoustic-Bass",
    33: "Electric-Bass-finger",
    34: "Electric-Bass-pick",
    35: "Fretless-Bass",
    36: "Slap-Bass-1",
    37: "Slap-Bass-2",
    38: "Synth-Bass-1",
    39: "Synth-Bass-2",

    # Strings & Orchestral
    40: "Violin",
    41: "Viola",
    42: "Cello",
    43: "Contrabass",
    44: "Tremolo-Strings",
    45: "Pizzicato-Strings",
    46: "Orchestral-Harp",
    47: "Timpani",

    # Ensemble
    48: "String-Ensemble-1",
    49: "String-Ensemble-2",
    50: "Synth-Strings-1",
    51: "Synth-Strings-2",
    52: "Choir-Aahs",
    53: "Voice-Oohs",
    54: "Synth-Voice",
    55: "Orchestra-Hit",

    # Brass
    56: "Trumpet",
    57: "Trombone",
    58: "Tuba",
    59: "Muted-Trumpet",
    60: "French-Horn",
    61: "Brass-Section",
    62: "Synth-Brass-1",
    63: "Synth-Brass-2",

    # Reeds
    64: "Soprano-Sax",
    65: "Alto-Sax",
    66: "Tenor-Sax",
    67: "Baritone-Sax",
    68: "Oboe",
    69: "English-Horn",
    70: "Bassoon",
    71: "Clarinet",

    # Pipes
    72: "Piccolo",
    73: "Flute",
    74: "Recorder",
    75: "Pan-Flute",
    76: "Blown-Bottle",
    77: "Shakuhachi",
    78: "Whistle",
    79: "Ocarina",

    # Synth Lead
    80: "Lead-1-square",
    81: "Lead-2-sawtooth",
    82: "Lead-3-calliope",
    83: "Lead-4-chiff",
    84: "Lead-5-charang",
    85: "Lead-6-voice",
    86: "Lead-7-fifths",
    87: "Lead-8-bass+lead",

    # Synth Pad
    88: "Pad-1-new-age",
    89: "Pad-2-warm",
    90: "Pad-3-polysynth",
    91: "Pad-4-choir",
    92: "Pad-5-bowed",
    93: "Pad-6-metallic",
    94: "Pad-7-halo",
    95: "Pad-8-sweep",

    # Effects
    96: "FX-1-rain",
    97: "FX-2-soundtrack",
    98: "FX-3-crystal",
    99: "FX-4-atmosphere",
    100: "FX-5-brightness",
    101: "FX-6-goblins",
    102: "FX-7-echoes",
    103: "FX-8-sci-fi",

    # Ethnic & Percussion
    104: "Sitar",
    105: "Banjo",
    106: "Shamisen",
    107: "Koto",
    108: "Kalimba",
    109: "Bag-pipe",
    110: "Fiddle",
    111: "Shanai",

    # Percussive
    112: "Tinkle-Bell",
    113: "Agogo",
    114: "Steel-Drums",
    115: "Woodblock",
    116: "Taiko-Drum",
    117: "Melodic-Tom",
    118: "Synth-Drum",
    119: "Reverse-Cymbal",

    # Sound Effects
    120: "Guitar-Fret-Noise",
    121: "Breath-Noise",
    122: "Seashore",
    123: "Bird-Tweet",
    124: "Telephone-Ring",
    125: "Helicopter",
    126: "Applause",
    127: "Gunshot"
}


REGULAR_NUM_DENOM = [(1, 1), (1, 2), (2, 2), (3, 2), (4, 2),
                     (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4),
                     (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (11, 8), (12, 8)]
CORE_NUM_DENOM = [(1, 1), (1, 2), (2, 2), (4, 2),
                  (1, 4), (2, 4), (3, 4), (4, 4), (5, 4),
                  (1, 8), (2, 8), (3, 8), (6, 8), (9, 8), (12, 8)]
VALID_TIME_SIGNATURES = ['time_signature_' + str(x[0]) + '/' + str(x[1]) for x in REGULAR_NUM_DENOM]

# cover possible time signatures
REGULAR_TICKS_PER_BEAT = [48, 96, 192, 384, 120, 240, 480, 960, 256, 512, 1024]
