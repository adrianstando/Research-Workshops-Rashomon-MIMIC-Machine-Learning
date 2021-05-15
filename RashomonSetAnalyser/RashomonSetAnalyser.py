class RashomonSetAnalyser:       
    
    def __init__(self):        
        self.base_model = None
        self.models = None
        self.rashomon_search_results = None
        self.model_profiles = None
        self.pdp_measures = None
    
    
    def set_base_model(self, base_model):
        """
        If you want to use created (and maybe fitted before) models, you can assign them to class attributes with this method.
        With this method you assign base model (the best one)
        
        argument: list: ['model_name', model_object] or model_object
        example: ['model_base', RandomForestClassifier(n_estimators = 30)] or  RandomForestClassifier(n_estimators = 30)
        """
        
        
        if(type(base_model)!=list):
            self.base_model=["model_base",base_model]
        else:
            self.base_model = base_model
    
    def set_models(self, models):
        """
        If you want to use created (and maybe fitted before) models, you can assign them to class attributes with this method.
        With this method you assign all models but the best one
        
        argument: list of such lists: ['model_name', model_object] or model_object
        example: [['model1', RandomForestClassifier(n_estimators = 10)], ['model2', RandomForestClassifier(n_estimators = 20)]] or  [RandomForestClassifier(n_estimators = 20), RandomForestClassifier(n_estimators = 30)]
        """
       
        self.models = models
        
        if(type(models[0])!=list):
            for i in range(len(models)):
                self.models[i]=(["model"+str(i),models[i]])
        
        
    def fit(self, X, y, *args, **kwargs):
        """
        Fits assigned models.
        """
        
        if self.base_model is None:
            raise Exception("Models were not chosen")
        
        self.base_model[1] = self.base_model[1].fit(X, y, *args, **kwargs)
        
        if self.models is None:
            return
        
        for i in range(len(self.models)):
            self.models[i][1] = self.models[i][1].fit(X, y, *args, **kwargs)
            
    
    def get_params(self):
        """
        Return dictionary of params of assigned models.
        """
        
        if self.base_model is None:
            raise Exception("Models were not created.")
        
        d = dict()
        d[self.base_model[0]] = self.base_model[1].get_params()
        
        if self.models is None:
            return d
        
        for model in self.models:
            d[model[0]] = model[1].get_params()
            
        return d
            
    
    def generate_rashomon_set(self, X, y, base_estimator, searcher_type = 'random', rashomon_ratio = 0.1, *args, **kwargs):
        """
        Searching for best models and choosing [rashomon_ratio %] best.
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        import pandas as pd
        import math
        import copy
        
        searcher_object = None
        
        if searcher_type == 'random':
            searcher_object = RandomizedSearchCV(base_estimator, *args, **kwargs)
        elif searcher_type == 'grid':
            searcher_object = GridSearchCV(base_estimator, *args, **kwargs)
        else:
            raise Exception("Wrong searcher type.")
            
        rashomon_ratio = min(max(0, rashomon_ratio), 1)      

        searcher_object.fit(X, y)
        results = pd.DataFrame(searcher_object.cv_results_).sort_values(by = 'mean_test_score', ascending = False).reset_index(drop=True)
        
        self.base_model = ["Base model", copy.deepcopy(base_estimator)]
        self.base_model[1].set_params(**results.params[0])
        
        n_models = min(max(math.floor(len(results.index) * rashomon_ratio), 1), len(results.index) - 1)
        self.models = []
        
        for i in range(n_models):
            m = copy.deepcopy(base_estimator)
            m.set_params(**results.params[i + 1])
            
            self.models.append(["Model " + str(i + 1), m])
            
        self.rashomon_search_results = results
        return results
    
    
    def change_rashomon_ratio(self, rashomon_ratio):
        """
        Changing rashomon ratio after generating set of models.
        """
        import math
        import copy
        
        if self.rashomon_search_results is None:
            raise Exception("Models were not generated. Run generate_rashomon_set method.")
        
        model = copy.deepcopy(self.base_model[1])
        n_models = min(max(math.floor(len(self.rashomon_search_results.index) * rashomon_ratio), 1), len(self.rashomon_search_results.index) - 1)
        self.models = []
        
        for i in range(n_models):
            m = copy.deepcopy(model)
            m.set_params(**self.rashomon_search_results.params[i + 1])
            
            self.models.append(["Model " + str(i + 1), m])
            
    @staticmethod
    def distance_function_generator(metric):
        import numpy as np
        from scipy.stats import kurtosis, skew
        
        if metric == 'abs_sum':
            return lambda x_base, y_base, x_new, y_new: np.sum(np.abs(y_base - y_new))
        elif metric == 'sum':
            return lambda x_base, y_base, x_new, y_new: np.sum(y_base - y_new)
        elif metric == 'integrate':
            return lambda x_base, y_base, x_new, y_new: np.sum((y_base - y_new) * x_new)
        elif metric == 'kurtosis':
            return lambda x_base, y_base, x_new, y_new: kurtosis(y_new)
        elif metric == 'skewness':
            return lambda x_base, y_base, x_new, y_new: skew(y_new)
        else:
            return lambda x_base, y_base, x_new, y_new: metric(x_base, y_base, x_new, y_new)
        
    
    def pdp_comparator(self, X, y, metric = 'abs_sum', save_model_profiles = False, variables = None, calculate_metric_for_base_model = False):
        """
        Compares pdp profiles with given metric.
        You can save (inside this object) model profiles from dalex if save_model_profiles set to True.
        If you set save_model_profiles=True, it requiers more memory, but you can calculate very fast different metrics with pdp_comparator_change_metric method.
        
        You can choose a certain subset of features by giving a list of these feature names as a variables parameter. If it's None, all features will be calculated.
        """
        import dalex as dx
        import pandas as pd
        import numpy as np
        
        distance = RashomonSetAnalyser.distance_function_generator(metric)
        
        profile_base = dx.Explainer(self.base_model[1], X, y, label = self.base_model[0], verbose = False)
        
        if variables is None:
            profile_base = profile_base.model_profile(verbose = False)
        else:
            profile_base = profile_base.model_profile(verbose = False, variables = variables)
        
        df = pd.DataFrame({'colname': profile_base.result._vname_.unique()})
        
        if save_model_profiles:
            self.model_profiles = [profile_base]
        
        y_base = profile_base.result._yhat_
        x_base = profile_base.result._x_
        
        sample_length = y_base.size / profile_base.result._vname_.nunique()
        
        for model in self.models:
            profile = dx.Explainer(model[1], X, y, label = model[0], verbose = False)
            
            if variables is None:
                profile = profile.model_profile(verbose = False)
            else:
                profile = profile.model_profile(verbose = False, variables = variables)
            
            y_result = profile.result._yhat_
            x_result = profile.result._x_
            
            tab_res = []
            for i in range(len(df.colname)):
                lower = int(i * sample_length)
                higher = int((i + 1) * sample_length)
                tab_res.append(distance(x_base[lower:higher], y_base[lower:higher], x_result[lower:higher], y_result[lower:higher]))
                
            df[model[0]] = tab_res
            
            if save_model_profiles:
                self.model_profiles.append(profile)
            else:
                del profile
                
        if calculate_metric_for_base_model:
            tab_res = []
            for i in range(len(df.colname)):
                lower = int(i * sample_length)
                higher = int((i + 1) * sample_length)
                tab_res.append(distance(x_base[lower:higher], y_base[lower:higher], x_base[lower:higher], y_base[lower:higher]))
            df[self.base_model[0]] = tab_res
        
        self.pdp_measures = df
        return df
    
        
    def pdp_comparator_change_metric(self, metric, calculate_metric_for_base_model = False):
        """
        You can use this method only if pdp_comparator was ran with parameter save_model_profiles=True
        It calculates results with new metric efficiently
        """
        import pandas as pd
        import numpy as np
        
        if self.model_profiles is None:
            raise Exception("Model profiles don't exist. Run pdp_comparator with parameter save_model_profiles = True to use this method.")
            
        distance = RashomonSetAnalyser.distance_function_generator(metric)
        
        profile_base = self.model_profiles[0]
        y_base = profile_base.result._yhat_
        x_base = profile_base.result._x_
        df = pd.DataFrame({'colname': profile_base.result._vname_.unique()})
        
        sample_length = y_base.size / profile_base.result._vname_.nunique()
        
        for j in range(1, len(self.model_profiles)):
            y_result = self.model_profiles[j].result._yhat_
            x_result = self.model_profiles[j].result._x_
            
            tab_res = []
            for i in range(len(df.colname)):
                lower = int(i * sample_length)
                higher = int((i + 1) * sample_length)
                tab_res.append(distance(x_base[lower:higher], y_base[lower:higher], x_result[lower:higher], y_result[lower:higher]))
            
            df[self.models[j - 1][0]] = tab_res
            
        if calculate_metric_for_base_model:
            tab_res = []
            for i in range(len(df.colname)):
                lower = int(i * sample_length)
                higher = int((i + 1) * sample_length)
                tab_res.append(distance(x_base[lower:higher], y_base[lower:higher], x_base[lower:higher], y_base[lower:higher]))
            df[self.base_model[0]] = tab_res
            
        self.pdp_measures = df
        return df
    
    
    def show_results(self, xlabels = False, model_names = None, features = None, figsize = (8, 8)):
        """
        You can use this method only if pdp_comparator was run.
        It generates barplots visualising results of comparator for all models.
        The method requires mtplotlib.pyplot installed
        
        If xlabels = True, the xlabels will be displayed
        You can choose models to be displayed by giving names in liast as model_names parameter
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        if self.pdp_measures is None:
            raise Exception("Model profiles don't exist. Run pdp_comparator first.")
            
        if features is None:
            features = self.pdp_measures.colname.tolist()
        
        labels = []
        if xlabels:
            labels = self.pdp_measures[self.pdp_measures['colname'].isin(features)].colname.tolist()
        
        if model_names is None:
            model_names = self.pdp_measures.columns.tolist()[1:]
                
        for i in model_names:
            fig, ax = plt.subplots(figsize = figsize)
            tmp = pd.DataFrame({'x': self.pdp_measures[self.pdp_measures['colname'].isin(features)].colname,
                               'y': self.pdp_measures[self.pdp_measures['colname'].isin(features)][i]})
            sns.barplot(data = tmp, x = 'x', y = 'y', color = 'dodgerblue')
            ax.set_xticklabels(labels)
            ax.set_title(i)
            ax.set_ylabel('')    
            ax.set_xlabel('')

            plt.show()

    def sum_plot(self, model_names = None, features = None, figsize = (8, 8)):
        """
        You can use this method only if pdp_comparator was run.
        It generates barplots visualising sums of values for each model returned by comparator.
        The method requires mtplotlib.pyplot installed
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        if self.pdp_measures is None:
            raise Exception("Model profiles don't exist. Run pdp_comparator first.")
        
        if features is None:
            features = self.pdp_measures.colname.tolist()
            
        sums = []
        
        if model_names is None:
            model_names = self.pdp_measures.columns.tolist()[1:]
            
        for i in model_names:
            sums.append(
                sum(
                    self.pdp_measures[self.pdp_measures['colname'].isin(features)][i]
                ))
        
        fig, ax = plt.subplots(figsize = figsize)
        tmp = pd.DataFrame({'x': model_names, 'sum': sums})
        sns.barplot(data = tmp, x = 'x', y = 'sum', color = 'dodgerblue')
        plt.title("Sum plot")
        ax.set_ylabel('')    
        ax.set_xlabel('')
        
        plt.show()
        
    def boxplots(self, xlabels = False, model_names = None, features = None, figsize = (8, 8)):
        """
        You can use this function to plot boxplot of results.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.pdp_measures is None:
            raise Exception("Model profiles don't exist. Run pdp_comparator first.")
            
        if model_names is None:
            model_names = self.pdp_measures.columns.tolist()[1:]
        
        if features is None:
            features = self.pdp_measures.colname.tolist()
        
        labels = []
        if xlabels:
            labels = self.pdp_measures[self.pdp_measures['colname'].isin(features)].colname.tolist()
            
        tdf = self.pdp_measures[self.pdp_measures['colname'].isin(features)][model_names + ['colname']].set_index('colname').T
        
        fig, ax = plt.subplots(figsize = figsize)
        sns.boxplot(data = tdf, color = 'dodgerblue')
        plt.title("Boxplots")
        ax.set_xticklabels(labels)
        ax.set_ylabel('')    
        ax.set_xlabel('')

        plt.show()
        
    def histograms(self, model_names = None, features = None, figsize = (8, 8)):
        """
        You can use this function to plot histograms of results.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        if self.pdp_measures is None:
            raise Exception("Model profiles don't exist. Run pdp_comparator first.")
            
        if model_names is None:
            model_names = self.pdp_measures.columns.tolist()[1:]
        
        if features is None:
            features = self.pdp_measures.colname.tolist()
            
        for i in model_names:
            fig, ax = plt.subplots(figsize = figsize)
            tmp = pd.DataFrame({'x': self.pdp_measures[self.pdp_measures['colname'].isin(features)].colname,
                               'y': self.pdp_measures[self.pdp_measures['colname'].isin(features)][i]})
            sns.histplot(data = tmp, x = 'y', color = 'dodgerblue')
            ax.set_title(i)
            ax.set_ylabel('')    
            ax.set_xlabel('')

            plt.show()