# Train-LightGBM-XGBoost-CatBoost-NGBoost-Based-GUI
快速选择最佳模型：轻松上手LightGBM、XGBoost、CatBoost和NGBoost！



[快速选择最佳模型：轻松上手LightGBM、XGBoost、CatBoost和NGBoost！ (qq.com)](https://mp.weixin.qq.com/s/JSoH6KwNBDEPgBW6FUeiwQ)



# 快速选择最佳模型：轻松上手LightGBM、XGBoost、CatBoost和NGBoost！

原创 Python机器学习AI [Python机器学习AI](javascript:void(0);) *2024年07月27日 00:29* *重庆*

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

***背景***

选择合适的模型通常需要理解数据集的特性、任务的目标以及模型的特点。然而，对于初学者或希望快速确定模型的用户来说，深入理解这些复杂的内容可能过于困难。因此，我们可以通过一个简化的GUI界面来进行模型选择。这个界面包括LightGBM、XGBoost、CatBoost和NGBoost这四个模型，并允许用户组合这些模型，通过赋予不同模型以不同的权重进行预测，可以形成大量的组合方式。为了简化这一过程，用户可以在默认参数下对模型进行组合，并通过评估指标选择最优的模型组合。然后，可以进一步调整这个最优组合的参数，以提升模型性能。这种方法无需用户深入理解每个模型的原理，就能实现高效的模型选择和优化，这个过程包括以下几个关键步骤：

***数据加载与预处理***

- 用户通过GUI加载数据，并选择要预测的目标变量，系统自动进行数据预处理，如数据集划分等

***模型选择与组合***

- 提供用户界面，让用户可以选择多个模型（LightGBM、XGBoost、CatBoost、NGBoost）以及它们的组合方式，组合方式可以是简单的平均，也可以是自定义权重

***默认参数下的模型评估***

- 系统使用默认参数对选择的模型进行训练和评估。评估指标如均方误差（MSE）、均方根误差（RMSE）、平均绝对误差（MAE）等，用于衡量模型性能
- 通过这种方式，可以快速比较不同模型或模型组合在默认参数下的表现

***选择最优模型组合***

- 根据评估结果，选择性能最好的模型或模型组合，这一步是自动化的，GUI会显示出最优组合和相应的评估指标

***进一步参数调整***

- 对选定的最优模型或模型组合进行超参数调整，以进一步提升性能，可以使用超参数优化方法如网格搜索、随机搜索或贝叶斯优化
- 调整后的模型性能再次进行评估，找到最优的参数设置

这个过程简化了模型选择和优化的复杂性，用户可以专注于数据的特性和任务目标，而无需深入理解每个模型的内部机制，这样的工具对于没有深厚技术背景的用户或希望快速原型开发的应用场景特别有用，示例应用场景

***代码***

```python

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ngboost import NGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics
import random
import time

# 定义平均模型类
class AverageModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)

    def fit(self, X, y, X_val, y_val):
        for model in self.models:
            if isinstance(model, lgb.LGBMRegressor):
                model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=100)])
            elif isinstance(model, xgb.XGBRegressor):
                model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            elif isinstance(model, CatBoostRegressor):
                model.fit(X, y, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
            elif isinstance(model, NGBRegressor):
                model.fit(X, y, X_val=X_val, Y_val=y_val)
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return np.average(predictions, axis=0, weights=self.weights)

# 初始化模型参数
params_lgb = {
    'learning_rate': 0.02,
    'boosting_type': 'gbdt',
    'objective': 'mse',
    'metric': 'rmse',
    'num_leaves': 127,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 4
}
model_lgb = lgb.LGBMRegressor(**params_lgb)

params_xgb = {
    'learning_rate': 0.02,
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'max_leaves': 127,
    'verbosity': 1,
    'seed': 42,
    'nthread': -1,
    'colsample_bytree': 0.6,
    'subsample': 0.7,
    'early_stopping_rounds': 100,
    'eval_metric': 'rmse'
}
model_xgb = xgb.XGBRegressor(**params_xgb)

params_cat = {
    'learning_rate': 0.02,
    'iterations': 1000,
    'depth': 6,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 100
}
model_cat = CatBoostRegressor(**params_cat)

params_ngb = {
    'learning_rate': 0.02,
    'n_estimators': 1000,
    'verbose': False,
    'random_state': 42,
    'natural_gradient': True
}
model_ngb = NGBRegressor(**params_ngb)

# GUI应用程序
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("机器学习模型预测")
        self.filepath = None
        self.models = []
        self.history = []
        self.weights = []

        self.model_names = ["LightGBM", "XGBoost", "CatBoost", "NGBoost"]
        self.model_entries = {}

        self.create_widgets()

    def create_widgets(self):
        self.label_file = tk.Label(self.root, text="选择文件:")
        self.label_file.grid(row=0, column=0, padx=10, pady=10)
        
        self.button_file = tk.Button(self.root, text="浏览", command=self.load_file)
        self.button_file.grid(row=0, column=1, padx=10, pady=10)
        
        self.label_target = tk.Label(self.root, text="选择目标变量:")
        self.label_target.grid(row=1, column=0, padx=10, pady=10)
        
        self.combo_target = ttk.Combobox(self.root)
        self.combo_target.grid(row=1, column=1, padx=10, pady=10)
        
        self.label_models = tk.Label(self.root, text="选择模型:")
        self.label_models.grid(row=2, column=0, padx=10, pady=10)
        
        self.var_lgb = tk.BooleanVar()
        self.check_lgb = tk.Checkbutton(self.root, text="LightGBM", variable=self.var_lgb)
        self.check_lgb.grid(row=2, column=1, sticky=tk.W)
        
        self.var_xgb = tk.BooleanVar()
        self.check_xgb = tk.Checkbutton(self.root, text="XGBoost", variable=self.var_xgb)
        self.check_xgb.grid(row=3, column=1, sticky=tk.W)
        
        self.var_cat = tk.BooleanVar()
        self.check_cat = tk.Checkbutton(self.root, text="CatBoost", variable=self.var_cat)
        self.check_cat.grid(row=4, column=1, sticky=tk.W)
        
        self.var_ngb = tk.BooleanVar()
        self.check_ngb = tk.Checkbutton(self.root, text="NGBoost", variable=self.var_ngb)
        self.check_ngb.grid(row=5, column=1, sticky=tk.W)
        
        self.button_random = tk.Button(self.root, text="随机选择模型", command=self.random_select_models)
        self.button_random.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        self.label_weighting = tk.Label(self.root, text="选择组合方式:")
        self.label_weighting.grid(row=7, column=0, padx=10, pady=10)
        
        self.combo_weighting = ttk.Combobox(self.root, values=["平均组合", "自定义权重"])
        self.combo_weighting.grid(row=7, column=1, padx=10, pady=10)
        self.combo_weighting.bind("<<ComboboxSelected>>", self.update_weight_entries)
        
        self.label_weights = tk.Label(self.root, text="输入权重:")
        self.label_weights.grid(row=8, column=0, padx=10, pady=10)
        
        for i, model_name in enumerate(self.model_names):
            tk.Label(self.root, text=f"{model_name}权重:").grid(row=8 + i, column=0, padx=10, pady=5)
            entry = tk.Entry(self.root, state=tk.DISABLED)
            entry.grid(row=8 + i, column=1, padx=10, pady=5)
            self.model_entries[model_name] = entry

        self.button_predict = tk.Button(self.root, text="预测", command=self.predict)
        self.button_predict.grid(row=13, column=0, columnspan=2, padx=10, pady=10)
        
        self.label_results = tk.Label(self.root, text="结果:")
        self.label_results.grid(row=14, column=0, padx=10, pady=10)
        
        self.text_results = tk.Text(self.root, height=10, width=50)
        self.text_results.grid(row=15, column=0, columnspan=2, padx=10, pady=10)

        self.button_history = tk.Button(self.root, text="显示历史模型", command=self.show_history)
        self.button_history.grid(row=16, column=0, columnspan=2, padx=10, pady=10)

    def load_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("Excel文件", "*.xlsx")])
        if self.filepath:
            try:
                self.df = pd.read_excel(self.filepath)
                self.combo_target['values'] = list(self.df.columns)
                messagebox.showinfo("成功", "文件加载成功。")
            except Exception as e:
                messagebox.showerror("错误", f"文件加载失败: {str(e)}\n请选择一个有效的Excel文件。")
    
    def random_select_models(self):
        self.var_lgb.set(False)
        self.var_xgb.set(False)
        self.var_cat.set(False)
        self.var_ngb.set(False)
        
        all_vars = [self.var_lgb, self.var_xgb, self.var_cat, self.var_ngb]
        selected_vars = random.sample(all_vars, random.randint(1, len(all_vars)))
        for var in selected_vars:
            var.set(True)

    def update_weight_entries(self, event):
        weighting_method = self.combo_weighting.get()
        if weighting_method == "自定义权重":
            for model_name, entry in self.model_entries.items():
                entry.config(state=tk.NORMAL)
        else:
            for model_name, entry in self.model_entries.items():
                entry.delete(0, tk.END)
                entry.config(state=tk.DISABLED)

    def predict(self):
        target = self.combo_target.get()
        if not target or not self.filepath:
            messagebox.showerror("错误", "请选择一个目标变量并加载数据。")
            return
        
        X = self.df.drop([target], axis=1)
        y = self.df[target]
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)
        
        x_mean = X_train.mean()
        x_std = X_train.std()
        y_mean = y.mean()
        y_std = y.std()
        
        X_train = (X_train - x_mean) / x_std
        y_train = (y_train - y_mean) / y_std
        X_val = (X_val - x_mean) / x_std
        y_val = (y_val - y_mean) / y_std
        X_test = (X_test - x_mean) / x_std
        y_test = (y_test - y_mean) / y_std
        
        self.models = []
        selected_models = []
        weights = []
        if self.var_lgb.get():
            self.models.append(model_lgb)
            selected_models.append("LightGBM")
        if self.var_xgb.get():
            self.models.append(model_xgb)
            selected_models.append("XGBoost")
        if self.var_cat.get():
            self.models.append(model_cat)
            selected_models.append("CatBoost")
        if self.var_ngb.get():
            self.models.append(model_ngb)
            selected_models.append("NGBoost")
        
        if len(self.models) == 0:
            messagebox.showerror("错误", "没有选择模型。")
            return
        
        if self.combo_weighting.get() == "自定义权重":
            try:
                weights = [float(self.model_entries[model_name].get()) for model_name in selected_models if self.model_entries[model_name].get()]
                if len(weights) != len(self.models) or not np.isclose(sum(weights), 1.0):
                    raise ValueError
            except ValueError:
                messagebox.showerror("错误", "请提供有效的权重，并确保权重之和为1。")
                return

        if not weights:
            weights = [1/len(self.models)] * len(self.models)
        
        average_model = AverageModel(self.models, weights=weights)
        start_time = time.time()
        average_model.fit(X_train, y_train, X_val, y_val)
        end_time = time.time()
        y_pred = average_model.predict(X_test)
        
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        
        result_text = (f"使用的模型: {', '.join(selected_models)}\n"
                       f"权重: {', '.join(map(str, weights))}\n"
                       f"训练时间: {end_time - start_time:.2f} 秒\n"
                       f"均方误差 (MSE): {mse}\n"
                       f"均方根误差 (RMSE): {rmse}\n"
                       f"平均绝对误差 (MAE): {mae}\n"
                       f"拟合优度 (R-squared): {r2}")
        self.text_results.insert(tk.END, result_text)
        
        # 保存历史记录
        self.history.append({
            'models': selected_models,
            'weights': weights,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

    def show_history(self):
        if not self.history:
            messagebox.showinfo("历史", "无历史记录。")
            return
        
        history_text = ""
        for i, entry in enumerate(self.history, 1):
            history_text += (f"运行 {i}:\n"
                             f"模型: {', '.join(entry['models'])}\n"
                             f"权重: {', '.join(map(str, entry['weights']))}\n"
                             f"均方误差 (MSE): {entry['mse']}\n"
                             f"均方根误差 (RMSE): {entry['rmse']}\n"
                             f"平均绝对误差 (MAE): {entry['mae']}\n"
                             f"拟合优度 (R-squared): {entry['r2']}\n\n")
        
        history_window = tk.Toplevel(self.root)
        history_window.title("模型历史记录")
        history_text_widget = tk.Text(history_window, height=20, width=60)
        history_text_widget.pack(padx=10, pady=10)
        history_text_widget.insert(tk.END, history_text)
        history_text_widget.config(state=tk.DISABLED)

# 运行GUI
root = tk.Tk()
app = App(root)
root.mainloop()
```

该代码由 ChatGPT 生成(当然不是一次性生成经过了不断优化迭代)，支持上述功能，通过以下链接获取AI网址：https://ai.zhangsan.cool/list （欲获取使用方法后文附有联系方式），该工具提供了一个GUI界面，用户可以通过选择LightGBM、XGBoost、CatBoost和NGBoost模型进行组合，快速评估它们在默认参数下的表现。用户可以根据不同组合形式下的模型评价指标选择最优的组合模型，对于进一步的参数调整，用户可以在确定模型后自行进行调参，运行代码后，将展示一个简洁易用的GUI界面供用户操作

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

接下来，点击“浏览”按钮导入需要进行预测的数据集。如果读取成功，系统会显示“文件加载成功”的提示。如果读取不成功，系统将返回“文件加载失败，请选择一个有效的Excel文件”提示

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

接下来，用户可以选择待预测的目标变量以及使用的模型，假设选择LightGBM单模型进行预测，系统会返回详细的评价指标，其中包括拟合优度，显示的结果为0.797595

![图片](https://mmbiz.qpic.cn/mmbiz_png/cCtGVD6h9mdzdqOXnWVHAGjicOc9pLmIogU8pdd6eJfqKUlW2TR3LGv0nnK9iaThpze6E3BUnNhIV5BNTVSoGCicA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

选择LightGBM和XGBoost的组合模型时，默认情况下，模型权重设置为平均组合，此时，系统会返回详细的评价指标，包括拟合优度，显示的结果为0.76075

![图片](https://mmbiz.qpic.cn/mmbiz_png/cCtGVD6h9mdzdqOXnWVHAGjicOc9pLmIoyL6wCibEPfltvUGuDljrPDPWFogZyv3pI3ElsJPdnwHv0wUJYMDxkLQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

接下来，我们对LightGBM和XGBoost的组合模型分配不同的权重，确保权重之和为1，经过调整后，系统返回的拟合优度为0.76938561

![图片](https://mmbiz.qpic.cn/mmbiz_png/cCtGVD6h9mdzdqOXnWVHAGjicOc9pLmIor1NNAQsuH8wroIFYhQA3tbTbq7JflspX2d39leQUSiauKsEicFGCWIYA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在已有模型的基础上，用户可以点击“显示历史模型”按钮，查看各种情况下的模型评价指标和模型组成。通过这种方式，可以进一步针对表现最好的模型进行参数调整，以提高其精确度，虽然此处仅展示了三种模型的组合，但实际上，可能的模型组合是无穷无尽的，这种方法帮助我们快速找到最适合当前数据的模型，为后续的处理和优化提供基础

![图片](https://mmbiz.qpic.cn/mmbiz_png/cCtGVD6h9mdzdqOXnWVHAGjicOc9pLmIoCjo0UH1ALQFsPun68Kia8mlO2Wy5wdbUVsSCTLExl6DzDSu7NnkKiaGg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

当然这只是一个比较简陋的GUI，可以对其进一步调整如自动化调参：进一步自动化超参数调优过程，模型扩展：支持更多模型类型或集成方法，用户友好性：改进GUI设计，增加可视化分析工具等

***往期推荐***

[K-means聚类与t-SNE降维：多维数据的二维可视化](http://mp.weixin.qq.com/s?__biz=Mzk0NDM4OTYyOQ==&mid=2247485877&idx=1&sn=d8006e72727ae220cba6ebc58ba754fc&chksm=c3242527f453ac31fe5eff87d7a20ff855b7e1ea81f0452dcd240c75d360ae56c0cef201842b&scene=21#wechat_redirect)

[小白轻松上手：一键生成SHAP解释图的GUI应用，支持多种梯度提升模型选择](http://mp.weixin.qq.com/s?__biz=Mzk0NDM4OTYyOQ==&mid=2247485757&idx=1&sn=0d6f837ab9f669d7262073e5467df09f&chksm=c32425aff453acb9b34cf795a9d21e42670d2b2a1560519c7829cacea8ec793fde09d654094d&scene=21#wechat_redirect)

[梯度提升集成：LightGBM与XGBoost组合预测](http://mp.weixin.qq.com/s?__biz=Mzk0NDM4OTYyOQ==&mid=2247484785&idx=1&sn=bbe0478110c9f13d6a03cad068b3e346&chksm=c32429e3f453a0f57f84e560ecaecf3467101b1f91e028f82cf0440db17cfcdd23ea81b36284&scene=21#wechat_redirect)

[特征工程进阶：暴力特征字典的构建与应用 实现模型精度质的飞跃](http://mp.weixin.qq.com/s?__biz=Mzk0NDM4OTYyOQ==&mid=2247485865&idx=1&sn=0466d4521df4867cc78e50be773f5b0d&chksm=c324253bf453ac2de23f999f7c022438234651127a5c370e8340c42f9a3d54b7c9cc342b77ce&scene=21#wechat_redirect)

[基于CatBoost回归预测模型的多种可解释性图表绘制](http://mp.weixin.qq.com/s?__biz=Mzk0NDM4OTYyOQ==&mid=2247485895&idx=1&sn=3a9a54ff95936d239f0c2bc770d14fa2&chksm=c3242555f453ac430489a40cc8d1fe24eb5c7eae34a06d0aafdd270f8ae78158df98a9c35eec&scene=21#wechat_redirect)

[无网络限制！同步官网所有功能！让编程小白也能轻松上手进行代码编写！！！](http://mp.weixin.qq.com/s?__biz=Mzk0NDM4OTYyOQ==&mid=2247485865&idx=2&sn=297411bc0e55ae2d8be65b0aaacd2a50&chksm=c324253bf453ac2d23eda28e223d20cd47ce8670534d4725e31e9efa1121acaef8cd2eebc56b&scene=21#wechat_redirect)

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

**微信号｜deep_ML
**

**欢迎添加作者微信进入Python、ChatGPT群**

**进群请备注Python或AI进入相关群
无需科学上网、同步官网所有功能、使用无限制**

![img](http://mmbiz.qpic.cn/mmbiz_png/cCtGVD6h9mfIQmYOfjvLAic35u0TA7AlVElAjOyt0pwcKBKibZScGyQDKIVlPmKT04xSlyuan4UibibibSKarGhQ5Ig/300?wx_fmt=png&wxfrom=19)

**Python机器学习AI**

常见数学模型、算法实战

100篇原创内容



公众号



如果你对类似于这样的文章感兴趣。

欢迎关注、点赞、转发~

*个人观点，仅供参考*

\##Python96

数据挖掘68

\#机器学习55

回归11

\##Python · 目录

上一篇基于CatBoost回归预测模型的多种可解释性图表绘制下一篇科研作图之同画布下的多重Y轴绘图



![img](https://mp.weixin.qq.com/mp/qrcode?scene=10000004&size=102&__biz=Mzk0NDM4OTYyOQ==&mid=2247485911&idx=1&sn=a3e7a0b44fb85c805f2829309f914e11&send_time=)

微信扫一扫
关注该公众号
