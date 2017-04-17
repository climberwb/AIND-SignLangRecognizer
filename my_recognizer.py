import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    probabilities = []
    guesses = []
    
    for s in range(0,test_set.num_items):
        X,lengths = test_set.get_item_Xlengths(s)
        
        sequences = test_set.get_item_sequences(s)
        gword,blogL = None,float('-inf')
        # word = test_set.df.iloc[s].word
        probs_dict={}
        
        for word,model in models.items():
            probs_dict[word]  = float('-inf') 
            try:
                probs_dict[word] = model.score(X,lengths) 
                if probs_dict[word] > blogL:
                    gword = word
                    blogL = probs_dict[word]
            except Exception as e:
                # print(str(e))
                pass
                    
        probabilities.append(probs_dict)    
        guesses.append(gword)
        
            # guesses.append(word)
            # probabilities.append(logL)
     # except:
     #  pass
    # def run_through_test_set(x):
    #  # print(x)
    #  pass
    #  # test_set.get_word_Xlengths(x.word)
     # model = models[x.word]
     # logL = model.score(x.get_item_sequences,x.get_item_Xlengths)
     # probabilities.append(logL)
     # guesses.append(x.word)
     # video speaker start_frame end_frame word
     # get_item_Xlengths
     # get_item_sequences
  
    # test_set.df.apply(run_through_test_set,axis=0)
    # TODO implement the recognizer
    return probabilities, guesses
    raise NotImplementedError
