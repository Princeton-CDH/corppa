# entity map based on the most common entities found in the corpus

entity_map = {
    "&indent;": "\t",
    "&yogh;": "ʒ",  # U+0292
    "&ast;": "*",
    "&wblank;": "\u2003",  # EM SPACE
    "&point;": ".",
    "&supere;": "ᵉ",  # U+1D49
    "&ebar;": "ē",  # U+0113
    "&obar;": "ō",  # U+014D
    "&supert;": "ᵗ",  # U+1D57
    # i'm not sure about the next one (https://www.compart.com/en/unicode/U+2423)
    # it might be a blank line instead of a space
    # tread cautiously, though, because it appears in metadata fields
    # if we'd just replace it with a blank line, we'd create a new row in the metadata...
    "&lblank;": "\u2423",  # long blank space?
    "&abar;": "ā",  # U+0101
    "&ubar;": "ū",  # U+016B
    "&superh;": "ʰ",  # U+02B0
    "&nbar;": "n\u0304",  # n + combining macron
    "&plus;": "+",  # plus sign
    "&superc;": "ᶜ",  # U+1D9C
    "&superr;": "ʳ",  # modifier letter small r (U+02B3)
    "&eshort;": "ĕ",  # e with breve (U+0115)
    "&ibar;": "ī",  # i with macron (U+012B)
    "&dollar;": "$",  # dollar sign
    "&equals;": "=",  # equals sign
    "&superu;": "ᵘ",  # U+1D52
    "&verbar;": "|",  # vertical bar
    "&supero;": "ᵒ",  # U+1D52
    "&mbar;": "m\u0304",  # m + combining macron
    "&superi;": "ⁱ",  # U+2071
    "&sblank;": " ",  # a blank?
    "&grn;": "ν",  # Greek nu
    "&sblankn;": "\u2009",  #  thin space/n-space? (U+2009)
    # "&sblankses;": "\u2002", # no idea! double space?
    "&ngr;": "ŋ",  # could be velar nasal
    "&igr;": "ι",  # iota?
    "&supers;": "ˢ",  # a superscript s
    "&es;": "es",  # this is "es" in the poem Ane Ballat of ye Captane of the Castell!
    "&YOGHantipus;": "ȝ",  # yogh?
    "&sblankr;": " ",  # regular blank space?
    "&sblankt;": " ",  # yet another blank variant...
    "&WBLANKR;": " ",  # a wide blank?
    "&SBLANKN;": " ",  # a small one now again? who knows...
    "&grst;": "σ",  # sigma (σ)
    "&sblanks;": " ",  # yay, another blank space marker
    "&pbar;": "ꝑ",  # “p–bar” is a medieval abbreviation (a p with a stroke, here U+A751)
    "&sfgr;": "φ",  # in a series of Greek abbreviations, “sfgr” is taken as phi (φ)
    "&YOGHuill;": "ȝ",  # another variant of yogh (ȝ)
    "&sblankd;": " ",  # blank (variant “d”)
    "&ishortness;": "",  # a “short” marker – used for abbreviation, here simply removed
    "&WBLANKTE;": " ",  # wide blank (TE)
    "&gria;": "α",  # in the Greek series, “gria” is best read as alpha (α)
    "&oshortries;": "ies",  # used to abbreviate an ending (as in “promontories” → “promont**&oshortries;**”)
    "&ishortrits;": "rit",  # an abbreviation marker (to complete, say, “spirit”)
    "&sblankhall;": " ",  # blank (variant “hall”)
    "&sblankzy;": " ",  # blank (variant “zy”)
    "&ashortel;": "",  # an “a–short” marker (for omitted letters; we drop it)
    "&ashortned;": "",  # likewise, a shortening marker
    "&amp;": "&",
    # i'm giving up after this -- the one's below are also mostly singletons!
    # "&sblankrts;": " ",
    # "&sblankb;":   " ",
    # "&sblanklla;": " ",
    # "&sblankmb;":  " ",
    # "&WBLANKD;":   " ",
    # "&ushorted;":  " ",
    # "&sblanke;":   " ",
    # "&SBLANKE;":   " ",
    # "&rbEEgr;":    "r",
    # "&ushortlus;": "",
    # "&sblankx;":   " ",
    # "&ishortl;":   "",
    # "&WBLANKE;":   " ",
    # "&hacekenquiry;": "",
    # "&ishortrit;": "",
    # "&sblankng;":  " ",
    # "&sblankry;":  " ",
    # "&sblankghs;": " ",
    # "&YOGHoe;":    "ȝ",
    # "&WBLANKGH;":  " ",
    # "&YOGHouthe;": "ȝ",
    "&gre;": "ε",  # ε
    # "&WBLANKSLY;": " ",
    # "&SBLANKM;":   " ",
    # "&sblankAthens;": " ",
    # "&sblankg;":   "",
    "&YOGH;": "ȝ",  # yogh
    "&gru;": "υ",  # upsilon (υ)
    # "&sblanklle;": "",
    "&Sgr;": "Σ",  # a capital Greek sigma (Σ)
    "&Igr;": "Ι",  # capital Greek iota (Ι)
    # "&ggriacgr;":  "",
    "&sblankse;": " ",  # blank???
}
