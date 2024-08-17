import numpy as np
class EIF:
    # 定义孤立数的一个节点，其中包含左右子树和决策信息及节点类型
    class node:
        def __init__(self, left=None, right=None, typeinfo='ex', normal_vector=None, intercept=None):
            self.left = left
            self.right = right
            self.type = typeinfo
            self.normal_vector = normal_vector
            self.intercept = intercept
    # 初始化EIF
    def __init__(self, n_estimators=300, sub_size=256, extension='full', contamination=0.1):
        self.height_limit = int(np.ceil(np.log2(sub_size)))
        self.n_estimators = n_estimators
        self.sub_size = sub_size
        self.extension = extension
        self.contamination = contamination
        self.git = True
    def get_np(self, data):
        dim = data.shape[1]
        flag = False
        # 自由度约束情况
        if type(self.extension) != str and self.extension < dim-1:
            zero_index = np.random.choice(range(dim), dim-(self.extension+1), replace=False)
            flag = True
        if flag:
            n = np.zeros(dim)
            for i in range(dim):
                if i not in zero_index:
                    n[i] = np.random.normal(0, 1, 1).item()
        else:
            n = np.random.normal(0, 1, dim)
        # 获取p
        area = [(data[:, col].min(), data[:, col].max()) for col in range(dim)]
        p = np.zeros(dim)
        for i, each in enumerate(area):
            p[i] = np.random.uniform(each[0], each[1], 1).item()
        return (n, p)
    def iTree(self, data, hight):
        if data.shape[0] <= 1 or hight == self.height_limit: # 外部节点情况
            return self.node(typeinfo='ex')
        else:
            # 获得n, p
            dim = data.shape[1]
            n, p = self.get_np(data)
            x_left = data[(data - p)@n <= 0]
            x_right = data[(data - p)@n > 0]
            params = dict(left=self.iTree(x_left, hight+1), right=self.iTree(x_right, hight+1), typeinfo='in'
                          , normal_vector=n, intercept=p)
            return self.node(**params)
                
    def fit(self, data):
        if data.shape[0] < 256:
            self.sub_size = int(data.shape[0] / 2)
        estimators = []
        for i in range(self.n_estimators):
            ind = np.random.choice(range(data.shape[0]), self.sub_size, replace=False) # 抽取sub_sample
            data_i = data[ind]
            estimators.append(self.iTree(data_i, 0))
        self.eif = estimators
        return self
    def c(self, size):
        return 2 * (np.log(size-1)+0.5772156649) - 2*(size-1)/size
    def path_length(self, sample, root, length):
        if root.type == 'ex':
            return length + self.c(self.sub_size)
        else:
            n, p = root.normal_vector, root.intercept
            if (sample - p) @ n <= 0:
                return self.path_length(sample, root.left, length+1)
            else:
                return self.path_length(sample, root.right, length+1)
    def decision_function(self, data):
        scores = np.zeros(data.shape[0]) # 所有样本分数
        for i in range(data.shape[0]):
            ensemble_score = np.zeros(len(self.eif)) # 单个样本的在森林每棵树的分数
            for j, tree in enumerate(self.eif):
                ensemble_score[j] = self.path_length(data[i], tree, 0)
            top, bottom = ensemble_score.mean(), self.c(self.sub_size)
            scores[i] = 2 ** ((top / bottom)*(-1))
        return scores
    def predict(self, data):
        scores = self.decision_function(data)
        threshold = np.percentile(scores, 100*(1-self.contamination))
        out = np.zeros(data.shape[0]).astype(int)
        out[scores >= threshold] = 1
        self.threshold = threshold
        return out