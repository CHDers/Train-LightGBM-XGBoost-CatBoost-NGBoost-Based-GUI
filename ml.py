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
                model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[
                          lgb.early_stopping(stopping_rounds=100)])
            elif isinstance(model, xgb.XGBRegressor):
                model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            elif isinstance(model, CatBoostRegressor):
                model.fit(X, y, eval_set=(X_val, y_val),
                          use_best_model=True, verbose=False)
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

        self.button_file = tk.Button(
            self.root, text="浏览", command=self.load_file)
        self.button_file.grid(row=0, column=1, padx=10, pady=10)

        self.label_target = tk.Label(self.root, text="选择目标变量:")
        self.label_target.grid(row=1, column=0, padx=10, pady=10)

        self.combo_target = ttk.Combobox(self.root)
        self.combo_target.grid(row=1, column=1, padx=10, pady=10)

        self.label_models = tk.Label(self.root, text="选择模型:")
        self.label_models.grid(row=2, column=0, padx=10, pady=10)

        self.var_lgb = tk.BooleanVar()
        self.check_lgb = tk.Checkbutton(
            self.root, text="LightGBM", variable=self.var_lgb)
        self.check_lgb.grid(row=2, column=1, sticky=tk.W)

        self.var_xgb = tk.BooleanVar()
        self.check_xgb = tk.Checkbutton(
            self.root, text="XGBoost", variable=self.var_xgb)
        self.check_xgb.grid(row=3, column=1, sticky=tk.W)

        self.var_cat = tk.BooleanVar()
        self.check_cat = tk.Checkbutton(
            self.root, text="CatBoost", variable=self.var_cat)
        self.check_cat.grid(row=4, column=1, sticky=tk.W)

        self.var_ngb = tk.BooleanVar()
        self.check_ngb = tk.Checkbutton(
            self.root, text="NGBoost", variable=self.var_ngb)
        self.check_ngb.grid(row=5, column=1, sticky=tk.W)

        self.button_random = tk.Button(
            self.root, text="随机选择模型", command=self.random_select_models)
        self.button_random.grid(
            row=6, column=0, columnspan=2, padx=10, pady=10)

        self.label_weighting = tk.Label(self.root, text="选择组合方式:")
        self.label_weighting.grid(row=7, column=0, padx=10, pady=10)

        self.combo_weighting = ttk.Combobox(
            self.root, values=["平均组合", "自定义权重"])
        self.combo_weighting.grid(row=7, column=1, padx=10, pady=10)
        self.combo_weighting.bind(
            "<<ComboboxSelected>>", self.update_weight_entries)

        self.label_weights = tk.Label(self.root, text="输入权重:")
        self.label_weights.grid(row=8, column=0, padx=10, pady=10)

        for i, model_name in enumerate(self.model_names):
            tk.Label(self.root, text=f"{model_name}权重:").grid(
                row=8 + i, column=0, padx=10, pady=5)
            entry = tk.Entry(self.root, state=tk.DISABLED)
            entry.grid(row=8 + i, column=1, padx=10, pady=5)
            self.model_entries[model_name] = entry

        self.button_predict = tk.Button(
            self.root, text="预测", command=self.predict)
        self.button_predict.grid(
            row=13, column=0, columnspan=2, padx=10, pady=10)

        self.label_results = tk.Label(self.root, text="结果:")
        self.label_results.grid(row=14, column=0, padx=10, pady=10)

        self.text_results = tk.Text(self.root, height=10, width=50)
        self.text_results.grid(
            row=15, column=0, columnspan=2, padx=10, pady=10)

        self.button_history = tk.Button(
            self.root, text="显示历史模型", command=self.show_history)
        self.button_history.grid(
            row=16, column=0, columnspan=2, padx=10, pady=10)

    def load_file(self):
        self.filepath = filedialog.askopenfilename(
            filetypes=[("Excel文件", "*.xlsx")])
        if self.filepath:
            try:
                self.df = pd.read_excel(self.filepath)
                self.combo_target['values'] = list(self.df.columns)
                messagebox.showinfo("成功", "文件加载成功。")
            except Exception as e:
                messagebox.showerror(
                    "错误", f"文件加载失败: {str(e)}\n请选择一个有效的Excel文件。")

    def random_select_models(self):
        self.var_lgb.set(False)
        self.var_xgb.set(False)
        self.var_cat.set(False)
        self.var_ngb.set(False)

        all_vars = [self.var_lgb, self.var_xgb, self.var_cat, self.var_ngb]
        selected_vars = random.sample(
            all_vars, random.randint(1, len(all_vars)))
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

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=42)

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
                weights = [float(self.model_entries[model_name].get(
                )) for model_name in selected_models if self.model_entries[model_name].get()]
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
