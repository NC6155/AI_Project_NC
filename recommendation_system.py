"""
Sistema de recomendación simple.
Uso:
 - load_data(path_csv)
 - model = train_model(df)
 - evaluate_model(model, df)
 - recommend(model, user_id, top_n)

Espera un CSV con columnas: user_id, item_id, rating
"""
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_data(path: str) -> pd.DataFrame:
	"""Carga CSV esperado con columnas user_id, item_id, rating"""
	df = pd.read_csv(path)
	required = {"user_id", "item_id", "rating"}
	if not required.issubset(df.columns):
		raise ValueError(f"CSV debe contener columnas: {required}")
	return df[["user_id", "item_id", "rating"]].dropna()


def _build_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
	users = df["user_id"].unique()
	items = df["item_id"].unique()
	user_idx = {u: i for i, u in enumerate(users)}
	item_idx = {i: j for j, i in enumerate(items)}
	mat = np.zeros((len(users), len(items)))
	for _, row in df.iterrows():
		ui = user_idx[row["user_id"]]
		ii = item_idx[row["item_id"]]
		mat[ui, ii] = row["rating"]
	return pd.DataFrame(mat, index=users, columns=items), user_idx, item_idx


def train_model(df: pd.DataFrame, n_components: int = 20, random_state: int = 42):
	"""Entrena descomposición SVD sobre la matriz usuario-item"""
	matrix, user_idx, item_idx = _build_matrix(df)
	svd = TruncatedSVD(n_components=min(n_components, min(matrix.shape)-1), random_state=random_state)
	user_factors = svd.fit_transform(matrix)
	item_factors = svd.components_.T
	model = {
		"svd": svd,
		"user_factors": user_factors,
		"item_factors": item_factors,
		"users": list(matrix.index),
		"items": list(matrix.columns),
		"user_idx": user_idx,
		"item_idx": item_idx,
		"matrix": matrix,
	}
	return model


def predict(model, user_id, item_id) -> float:
	"""Predice rating para user_id y item_id"""
	if user_id not in model["users"] or item_id not in model["items"]:
		return 0.0
	ui = model["users"].index(user_id)
	ii = model["items"].index(item_id)
	pred = float(np.dot(model["user_factors"][ui], model["item_factors"][ii]))
	return pred


def recommend(model, user_id, top_n: int = 10) -> List[Tuple[str, float]]:
	"""Devuelve top_n items recomendados (item_id, score)"""
	if user_id not in model["users"]:
		return []
	ui = model["users"].index(user_id)
	scores = model["item_factors"].dot(model["user_factors"][ui])
	seen = set(model["matrix"].columns[model["matrix"].loc[user_id] > 0])
	candidates = [(item, float(score)) for item, score in zip(model["items"], scores) if item not in seen]
	candidates.sort(key=lambda x: x[1], reverse=True)
	return candidates[:top_n]


def evaluate_model(model, df: pd.DataFrame) -> dict:
	"""Evalúa RMSE sobre un 20% de holdout"""
	train, test = train_test_split(df, test_size=0.2, random_state=42)
	model_train = train_model(train)
	preds = []
	trues = []
	for _, row in test.iterrows():
		trues.append(row["rating"])
		preds.append(predict(model_train, row["user_id"], row["item_id"]))
	rmse = mean_squared_error(trues, preds, squared=False)
	return {"rmse": rmse}


if __name__ == "__main__":
	import argparse

	p = argparse.ArgumentParser()
	p.add_argument("csv", help="Ruta al CSV con columnas user_id,item_id,rating")
	p.add_argument("--user", help="user_id para recomendar", default=None)
	p.add_argument("--top", help="top N", type=int, default=10)
	args = p.parse_args()
	df = load_data(args.csv)
	model = train_model(df)
	metrics = evaluate_model(model, df)
	print(f"RMSE: {metrics['rmse']:.4f}")
	if args.user:
		recs = recommend(model, args.user, top_n=args.top)
		for item, score in recs:
			print(f"{item}\t{score:.4f}")

