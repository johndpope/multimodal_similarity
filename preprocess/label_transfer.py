# label version for NIPS experiments, 04/25/2018

MIN_LENGTH = 5    # remove short sequences
MIN_LENGTH_BACKGROUND = 15    # remove short background sequences
MAX_LENGTH = 45    # maximum sequence length

label_transfer = {0: 0,    # background,
                  1: 1,    # intersection passing
                  2: 2,    # left turn
                  3: 3,    # right turn
                  4: 4,    # left lane change
                  5: 5,    # right lane change
                  6: 1,    # crosswalk passing --> intersection passing
                  7: 6,    # U-turn
                  8: 4,    # left lane branch --> left lane change
                  9: 5,    # right lane branch --> right lane change
                  10: 0    # merge --> background
                  }


honda_num2labels = {0: 'Background',
                    1: 'Intersection passing',
                    2: 'Left turn',
                    3: 'Right turn',
                    4: 'Left lane change',
                    5: 'Right lane change',
                    6: 'U-turn'
                    }
