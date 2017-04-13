import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # p is the number of parameters,
        # and N is the number of data points
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        hmm_model = None
        bics = []
        for num_states in range(self.min_n_components,self.max_n_components+1):
            # TODO
            # 1. implement create cv loop for test and train
            # 2. find average log Likelihood of cross validation fold
            # 3. pick the highest scoring model
            split_method = KFold()
            # print(self.sequences)
            try:
                av_bic=[]
                if(len(self.sequences)>2):
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences )
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        # print(X_test, lengths_test)
                else:
                    
                    X_train, lengths_train = self.X[:self.lengths[0]],[self.lengths[0]]
                    X_test, lengths_test = self.X[:self.lengths[1]],[self.lengths[1]]
                    
            
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                logL = hmm_model.score(X_test,lengths_test)
                BIC = -2 * logL + num_states * np.log(sum(lengths_train))
                
                av_bic.append(BIC)
                av = np.mean(av_bic)
                bics.append((int(av),hmm_model))
                   
            except  Exception as e:
                pass
        if bics==[]:
            return None
        smallest_bic = min(bics,key=lambda x: x[0])

        return smallest_bic[1]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        # warnings.filterwarnings("ignore", category=DeprecationWarning)
        hmm_model = None
        logs = []
        for num_states in range(self.min_n_components,self.max_n_components+1):
            # TODO
            # 1. implement create cv loop for test and train
            # 2. find average log Likelihood of cross validation fold
            # 3. pick the highest scoring model
            split_method = KFold()
            # print(self.sequences)
            try:
                av_log=[]
                if(len(self.sequences)>2):
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences )
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        # print(X_test, lengths_test)
                else:
                    
                    X_train, lengths_train = self.X[:self.lengths[0]],[self.lengths[0]]
                    X_test, lengths_test = self.X[:self.lengths[1]],[self.lengths[1]]
                    
            
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                logL = hmm_model.score(X_test,lengths_test)
                av_log.append(logL)
                av = np.mean(av_log)
                logs.append((int(av),hmm_model))
                   
            except  Exception as e:
                pass
        if logs==[]:
            return None
        largest_log = max(logs,key=lambda x: x[0])
        # print(smallest_log,[log[0]for log in logs],sorted([log[0]for log in logs]))
        # for l in logs:
        #     print(l[0],smallest_log)
        #     if l[0]== smallest_log:
        #         return l[1]
        return largest_log[1]
                # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx)) 
        # TODO implement model selection using CV
        #raise NotImplementedError
        
