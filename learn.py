import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RESULTADO_BINS = 30

debug = False
if len(sys.argv) == 2:
	print("[INFO] Debugging on")
	debug = True

def is_float_str(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def main():
	df = pd.read_csv("compiled_data.csv")
 
	dfResultadoNum = df[pd.to_numeric(df['Resultado'], errors='coerce').notna()]
	dfResultadoNum["ResultadoParsed"] = dfResultadoNum["Resultado"].apply(lambda x: float(x))
	dfResultadoNum["ResultadoLabels"] = pd.cut(dfResultadoNum["ResultadoParsed"], bins=RESULTADO_BINS)
	dfResultadoNum = dfResultadoNum.drop(["ResultadoParsed"], axis=1)
 
	dfResultadoStr = df[pd.to_numeric(df['Resultado'], errors='coerce').isna()]
	dfResultadoStr["ResultadoLabels"] = dfResultadoStr["Resultado"]
	
	df = pd.concat([dfResultadoStr, dfResultadoNum])
	df = df.drop(["Resultado"], axis=1)
 
	onehot_gp = pd.get_dummies(df['GrupoProduto'], prefix='GrupoProduto')
	onehot_produto = pd.get_dummies(df['Produto'], prefix='Produto')
	onehot_distr = pd.get_dummies(df['Distribuidora'], prefix='Distribuidora')
	onehot_uf = pd.get_dummies(df['Uf'], prefix='Uf')
	onehot_regiao = pd.get_dummies(df['RegiaoPolitica'], prefix='RegiaoPolitica')
	onehot_ensaio = pd.get_dummies(df['Ensaio'], prefix='Ensaio')
	onehot_resultado = pd.get_dummies(df['ResultadoLabels'], prefix='Resultado')
	df = pd.concat([df, onehot_gp, onehot_produto, onehot_distr, onehot_uf, onehot_regiao, onehot_ensaio, onehot_resultado], axis=1)
 
	df = df.drop(columns=["GrupoProduto", "Produto", "Distribuidora", "Uf", "RegiaoPolitica", "Ensaio", "UnidadeEnsaio", "ResultadoLabels"], axis=1)
	df = df.drop(columns=["Endereço", "Complemento", "Bairro", "Município", "Latitude", "Longitude", "RazaoSocialPosto", "CnpjPosto"], axis=1)
	df = df.drop(columns=["DataColeta", "IdNumeric"], axis=1)
	# print(df.head())

	# features = ["GrupoProduto", "Produto", "RazaoSocialPosto", "Distribuidora", "UF", "RegiaoPolitica", "Ensaio", "Resultado", "UnidadeEnsaio"]
	# features = ["GrupoProduto", "Produto", "Distribuidora", "Uf", "RegiaoPolitica", "Ensaio", "Resultado", "UnidadeEnsaio"]
	features = [col for col in df.columns if col != 'Conforme']
	X = df[features]
	y = df["Conforme"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	classifier = RandomForestClassifier()
	classifier.fit(X_train, y_train)
	predictions = classifier.predict(X_test)

	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy:", accuracy)

	return 0

if __name__ == "__main__":
	main()