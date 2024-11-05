#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Curso Machine Learning, herramientas para clasificar.

SVM , LR , KNN , DT , tal vez otro...


Created on Mon Oct 21 14:26:52 2024

@author: mariano
"""
#importo libraries que vamos a usar
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

df = pd.read_csv('/home/mariano/Documents/tomate/Congresos/A2B2C2023/2024/CursoML/penguins_size.csv')

df.head(5)#para ver las primeras 5 lineas del dataset

df.columns
#ahora vemos que el dataset tiene las columnas
#especie isla longitud_culmen produndiad_culmen
#longitud_aleta masa_corporal sexo

#vamos a verificar algunas caracteristicas, para ello, necesitamos
#hablar de estadistica... vamos a hablar de las medidas de resumen

#media, mediana, quartiles, max, min

#como diria jack el destripador, vamos por partes...

#armemos un sub df con fines puramente didacticos

Adelie_macho = df[(df['species']=='Adelie') & (df['sex']=='MALE')]

media_masa = Adelie_macho['body_mass_g'].mean()

mediana_masa = Adelie_macho['body_mass_g'].median()

cuantil_50_masa = Adelie_macho['body_mass_g'].quantile(0.5)

cuantil_25_masa = Adelie_macho['body_mass_g'].quantile(0.25)

cuantil_75_masa = Adelie_macho['body_mass_g'].quantile(0.75)

desvio = Adelie_macho['body_mass_g'].std()

print(f'media: {media_masa} mediana: {mediana_masa} quantil50: {cuantil_50_masa} quantil25: {cuantil_25_masa} quantil75: {cuantil_75_masa} desvio: {desvio}')

#sns.kdeplot(data = Adelie_macho , x = 'body_mass_g')


#el grafiquito...

Adelie_macho['body_mass_g'].plot.kde().set(title='Distribucion peso de machos Adelie', xlabel='Peso(g)')

plt.axvline(x = media_masa, color='red')

plt.axvline(x = mediana_masa, color = 'blue')

plt.axvline(x = cuantil_50_masa,color='yellow', linestyle='--')

plt.axvline(x = cuantil_25_masa, color='orange')

plt.axvline(x = cuantil_75_masa, color='violet')

plt.axvline(x = media_masa+desvio , color='black' , linestyle = '-.')

plt.axvline(x = media_masa-desvio , color='black' , linestyle = '-.')

plt.show()

plt.close()

#quiero agregarle a ese grafico constantes en x como la media, mediana

#como metodo super general existe describe

df.describe()

#por ejemplo...cuantos pinguinos hay por isla??

df['island'].value_counts()

#todo muy lindo, pero esto es una tabla, y gr;aficamente? como podria
#verlo?  una buena herramienta son los graficos de barras

df['island'].value_counts().plot.bar().set(title='Pinguinos por isla')
plt.show()
plt.close()

#hagan una tabla, y posteriormente un grafico de numero de 
#individuos por especie

Adelie_macho['body_mass_g'].plot.kde()

#podriamos preguntarnos, cual es la distribucion de los pesos de los
#pinguinos?

#jugar con los bins, que pasa si ponen menos, y si ponen mas?

#y si los bins tienden a infinito??
#es decir, si el bin tiene a agarrar solo u valor??

#df.plot.hist(bins = )

df['body_mass_g'].plot(kind = 'hist').set(xlabel = 'masa(g)')
plt.show()
plt.close()



df['body_mass_g'].plot(kind = 'kde').set(xlabel = 'masa(g)')#density kernel
plt.show()
plt.close()

#pareciera tener varios picos, valdra la pena preguntarse por especie?

#probemos graficando con una herramienta mas poderosa, y versatil
#que nos permite especificar con ciertos parametros, lo que queremos
#graficar, y nos va a ser muy util.

sns.kdeplot(data = df[df['species'] == 'Gentoo'] , x = 'body_mass_g' , hue = 'sex').set(xlabel = 'masa(g)')
plt.show()
plt.close()

#####clase sns viejo####
sns.kdeplot(x = df['body_mass_g'], hue = df['species']).set(xlabel = 'masa(g)')
plt.show()
plt.close()
######
#ahora podemos ver que en realidad la dsitribucion de pesos
#de cada especie es bien dsitinta, no?

#por que no nos disponems a ver si
# en alguna especie, y vemos que pasa con los sexos, seran distintos?

sns.violinplot(data = df ,
               x = 'species' , y = 'body_mass_g' , hue = 'sex').set(xlabel = 'especie' , ylabel='masa(g)')
plt.show()
plt.close()

#de aca logramos desprender que usando dos variables categoricas
#para separar la visualizacion de datos, y una numerica continua,
#que efectivamente el dataset tiene poblaciones distintas
#observando el peso, entre machos , y hembras, para cada especie
#ademas se observa que en el sexado hay elementos faltantes

#ya si quisieramos ver relaciones entre variables, puede resultar
#conveniente ir a otro tipo de graficos, como los de dispercion
#o scatter plot...


adelie = df[(df['species']=='Adelie')]


sns.kdeplot(df = adelie , x = ''body_mass_g' , hue = 'sex')

sns.scatterplot(data = df , x = 'flipper_length_mm', y = 'body_mass_g').set(xlabel = 'Largo aleta(cm)' , ylabel = 'masa(g)')
plt.show()
plt.close()
#parece una buena relacion, pero fue una buena idea no diferenciar por
#sexo? o por especie?
#veamos...

sns.scatterplot(data = df , x = 'flipper_length_mm', y = 'body_mass_g' , hue = 'species' , style = 'sex').set(xlabel = 'Largo aleta(cm)' , ylabel = 'masa(g)')
plt.show()
plt.close()

sns.scatterplot(data = df , x = 'flipper_length_mm', y = 'body_mass_g' , hue = (df[['species','sex']].apply(tuple, axis=1))).set(xlabel = 'Largo aleta(cm)' , ylabel = 'masa(g)')
plt.show()
plt.close()
#con esta visualizacion, podemos ver hasta 4 variables en juego
#en un grafico 2D

#bueno, por loque se ve, una recta ajustaria bastante bien a esos datos, no?

#probemos graficar con un ajuste lineal, a ver que pinta tienen...

sns.lmplot(data=df, x='flipper_length_mm', y='body_mass_g').set(title = 'regresion lineal Largo aleta(cm) vs masa(g)')
plt.show()
plt.close()

#ah, y si mejor separamos por sexos??

sns.lmplot(data=df, x='flipper_length_mm', y='body_mass_g', hue='sex').set(title = 'regresion lineal Largo aleta(cm) vs masa(g)')
plt.show()
plt.close()

#les dejo la tarea de agarrar a cada especie,
# y graficar rectas separando por sexo...


#y si quisiera ver las relaciones entre multiples variables?
#de una sola vez?
#Bueno!!!  existe una herrmaienta para ello...

sns.pairplot(data = df)

#aca vemos que hace scatter fuera de la diagonal, y en esta, hist
#ven que fuera del a diagonal, la informacion es redundante??

#habra manera de aprobechar esto??

#bueno, la hay!!

plot = sns.PairGrid(df)

plot.map_diag(sns.histplot)#histograma en la diagonal

plot.map_upper(sns.scatterplot)

plot.map_lower(sns.kdeplot)

####y diferenciados por especie?? funcionara???

sns.pairplot(data = df , hue = 'species', style = 'sex')

#aca vemos que hace scatter fuera de la diagonal, y en esta, hist
#ven que fuera del a diagonal, la informacion es redundante??

#habra manera de aprobechar esto??

#bueno, la hay!!

plot = sns.PairGrid(data = df , hue = 'species')

plot.map_diag(sns.histplot)#histograma en la diagonal

plot.map_upper(sns.scatterplot)

plot.map_lower(sns.kdeplot)

#podriamso querer ver con numeros, si existen
#correlaciones entre las distintas variables

correlacion = df.corr()

#y como verlo? bueno, para este tipo de cosas estan los heatmap

sns.heatmap(correlacion)

#y si ademas quisiera ver relaciones jerarquicas entre variables?

sns.clustermap(correlacion)

#hay algun metodo para agrupar-ver si las especies son diferenciables
#facil-rico-barato? en solo dos dimensiones, teniendo en cuenta
#varias variables a la vez???

#bueno, si, es un metodo llamado PCA!!!

from sklearn.decomposition import PCA

import numpy as np

#voy a aplicar el metodo solo a un subconjunto de los datos,
#por simplicidad, solo me quedo con machos de las tres especies


###aca una reducicionde la dimencionalidad, a ver si conserva 
#cierta capacidad de discriminar

X = df[(df['sex']=='MALE')].loc[: , ['culmen_length_mm',
'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]

pca = PCA(n_components=2)

matriz_pca = pca.fit_transform(X)

pd.DataFrame(matriz_pca).plot(x = 0 , y = 1 , kind = 'scatter').set(xlabel = 'Componente 1' , ylabel = 'Componente 2')

######aca si se animan, agreguenle a este dataset especie y sexo
#y evaluen si siguen siendo separables o no



#################aca una suerte de clustering hecho a mano

X = df[(df['sex']=='MALE')].loc[: , ['culmen_length_mm',
'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]

pca = PCA(n_components=2)

matriz_pca = pca.fit_transform(X.T)

pd.DataFrame(matriz_pca).plot(x = 0 , y=1 , kind = 'scatter').set(xlabel = 'Componente 1' , ylabel = 'Componente 2')

#####guia de ejercicios

##########################################################33

#spoiler!!!!!

########################################################33333

#pinguinos por isla

por_isla = df.groupby(['island','species']).size().reset_index(name='count')

sns.barplot(data = por_isla , x = 'island' , y = 'count' , hue = 'species')

##########hasta aca venia algo medio descriptivo...

####ahora se vienen modelos para clasificar posta...

#Arrancamos con LR

#mi objetivo, es sexar pinguinos de una dada especie, mediante el uso de regresion logistica

#separo un df por especie...

adelie = df[(df['species']=='Adelie')].loc[: , ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 
'body_mass_g', 'sex']]

chin = df[(df['species']=='Chinstrap')].loc[: , ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 
'body_mass_g', 'sex']]

gentoo = df[(df['species']=='Gentoo')].loc[: , ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 
'body_mass_g', 'sex']]

#primero cambio los valores, para pdoer usar regresion logistica

#la funcion CerosUnos(dataframe , columna) toma un df y una col, y le aplica a esa col la conversion de
#female a 1 , y de male a 0

def CerosUnos(dataframe , columna):
    
    dataframe[columna] = dataframe[columna].map({'FEMALE':1 , 'MALE':0})
    
    return(dataframe)

CerosUnos(chin , 'sex')

#chequeo
chin['sex']

####genero otra funcion que calcula el ratio entre dos variables, y genera una
#nueva columna con dicho ratio

def ratio(dataframe , columna1 , columna2):
    
    dataframe['culmen_ratio'] = dataframe[columna1]/dataframe[columna2]
    
    return(dataframe)

ratio(chin , 'culmen_length_mm' , 'culmen_depth_mm')

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

# la idea es usar esto... fpr, tpr, thresholds = metrics.roc_curve(y, scores)
#oara graficar curva roc es tpr vs fpr
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(

chin.loc[: , ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g' , 'culmen_ratio']] ,
chin.loc[:,['sex']],

test_size = 0.2 , random_state = 524)

print(X_train.shape , X_test.shape , y_train.shape , y_test.shape)

#pienso hacer una regresion logistica para culmen_length , culmen depth , flipper_ length , body_mass ,
#culmen_ratio , por separado, y luego tomar alguna desicion

from sklearn.linear_model import LogisticRegression

reg_largo_pico = LogisticRegression()

reg_profundidad_pico = LogisticRegression()

reg_ratio_pico = LogisticRegression()

reg_largo_aleta = LogisticRegression()

reg_masa = LogisticRegression()

reg_largo_pico.fit(np.asarray(X_train['culmen_length_mm']).reshape(-1 , 1) , np.asarray(y_train).ravel())

reg_profundidad_pico.fit(np.asarray(X_train['culmen_depth_mm']).reshape(-1 , 1) , np.asarray(y_train).ravel())

reg_ratio_pico.fit(np.asarray(X_train['culmen_ratio']).reshape(-1 , 1) , np.asarray(y_train).ravel())

reg_largo_aleta.fit(np.asarray(X_train['flipper_length_mm']).reshape(-1 , 1) , np.asarray(y_train).ravel())

reg_masa.fit(np.asarray(X_train['body_mass_g']).reshape(-1 , 1) , np.asarray(y_train).ravel())
#hasta aca fitie todos los modelos, queda usarlos, y graficar

#entonces ahora, ya puedo predecir las probabilidades, para graficar un poco

largo_pico_proba = reg_largo_pico.predict_proba(np.asarray(X_test['culmen_length_mm']).reshape(-1 , 1))

profundidad_pico_proba = reg_profundidad_pico.predict_proba(np.asarray(X_test['culmen_depth_mm']).reshape(-1 , 1))

ratio_pico_proba = reg_ratio_pico.predict_proba(np.asarray(X_test['culmen_ratio']).reshape(-1 , 1))

largo_aleta_proba = reg_largo_aleta.predict_proba(np.asarray(X_test['flipper_length_mm']).reshape(-1 , 1))

masa_proba = reg_masa.predict_proba(np.asarray(X_test['body_mass_g']).reshape(-1 , 1))

#####grafico las probabilidades, a ver si hay alguna diferencia...
####cueationable, estaria mejor hacer las curvas roc
plt.plot(np.asarray(X_test['culmen_length_mm']).reshape(-1 , 1), largo_pico_proba[:, 1], "g-", label="Female")
plt.plot(np.asarray(X_test['culmen_length_mm']).reshape(-1 , 1), largo_pico_proba[:, 0], "b--", label="Male")
plt.plot(y_test , ".")

####
plt.plot(np.asarray(X_test['culmen_depth_mm']).reshape(-1 , 1), profundidad_pico_proba[:, 1], "g-", label="Female")
plt.plot(np.asarray(X_test['culmen_depth_mm']).reshape(-1 , 1), profundidad_pico_proba[:, 0], "b--", label="Male")
plt.plot(y_test , ".")

###
plt.plot(np.asarray(X_test['culmen_ratio']).reshape(-1 , 1), ratio_pico_proba[:, 1], "g-", label="Female")
plt.plot(np.asarray(X_test['culmen_ratio']).reshape(-1 , 1), ratio_pico_proba[:, 0], "b--", label="Male")
plt.plot(y_test , ".")

###
plt.plot(np.asarray(X_test['flipper_length_mm']).reshape(-1 , 1), largo_aleta_proba[:, 1], "g-", label="Female")
plt.plot(np.asarray(X_test['flipper_length_mm']).reshape(-1 , 1), largo_aleta_proba[:, 0], "b--", label="Male")
plt.plot(y_test , ".")

####
plt.plot(np.asarray(X_test['body_mass_g']).reshape(-1 , 1), masa_proba[:, 1], "g-", label="Female")
plt.plot(np.asarray(X_test['body_mass_g']).reshape(-1 , 1), masa_proba[:, 0], "b--", label="Male")
plt.plot(y_test , ".")

####

#otra cosa que puedo hacer es graficar los preditcc, con los puntos originales, a ver como sale eso..

#usando .predict

largo_pico_pred = reg_largo_pico.predict(np.asarray(X_test['culmen_length_mm']).reshape(-1 , 1))

profundidad_pico_pred = reg_profundidad_pico.predict(np.asarray(X_test['culmen_depth_mm']).reshape(-1 , 1))

ratio_pico_pred = reg_ratio_pico.predict(np.asarray(X_test['culmen_ratio']).reshape(-1 , 1))

largo_aleta_pred = reg_largo_aleta.predict(np.asarray(X_test['flipper_length_mm']).reshape(-1 , 1))

masa_pred = reg_masa.predict(np.asarray(X_test['body_mass_g']).reshape(-1 , 1))

#graficamos...

plt.plot(np.asarray(X_test['culmen_length_mm']).reshape(-1 , 1), largo_pico_pred , "g.", label="Female")
#plt.plot(np.asarray(X_test['culmen_length_mm']).reshape(-1 , 1), largo_pico_pred[:, 0], "b--", label="Male")
plt.plot(y_test , "b.")

###
plt.plot(np.asarray(X_test['culmen_depth_mm']).reshape(-1 , 1), profundidad_pico_pred , "g.")
#plt.plot(np.asarray(X_test['culmen_length_mm']).reshape(-1 , 1), largo_pico_pred[:, 0], "b--", label="Male")
plt.plot(y_test , "b.")

###
plt.plot(np.asarray(X_test['flipper_length_mm']).reshape(-1 , 1), largo_pico_pred , "g.")
#plt.plot(np.asarray(X_test['culmen_length_mm']).reshape(-1 , 1), largo_pico_pred[:, 0], "b--", label="Male")
plt.plot(y_test , "b.")

###
#ahora chequeamos que los scores sean buenos

score_long_pico = reg_largo_pico.score(np.asarray(X_test['culmen_length_mm']).reshape(-1 , 1) , np.asarray(y_test).reshape(-1,1))

score_prof_pico = reg_profundidad_pico.score(np.asarray(X_test['culmen_depth_mm']).reshape(-1 , 1) , np.asarray(y_test).reshape(-1,1))

score_ratio_pico = reg_ratio_pico.score(np.asarray(X_test['culmen_ratio']).reshape(-1 , 1) , np.asarray(y_test).reshape(-1,1))

score_aleta = reg_largo_aleta.score(np.asarray(X_test['flipper_length_mm']).reshape(-1 , 1) , np.asarray(y_test).reshape(-1,1))

score_masa = reg_masa.score(np.asarray(X_test['body_mass_g']).reshape(-1 , 1) ,np.asarray(y_test).reshape(-1,1))

lista_score = [score_long_pico , score_prof_pico , score_ratio_pico , score_aleta , score_masa]

lista_barras = ['long pico' , 'prof pico' , 'ratio pico' , 'aleta' , 'masa']

sns.barplot(x = lista_score , y = lista_barras)

###a juzgar por score, solo vale la pena usar 'culmen_depth_mm' , 'culmen_length_mm'

####hacer curva roc

## la idea es usar esto... fpr, tpr, thresholds = metrics.roc_curve(y, scores)
#oara graficar curva roc es tpr vs fpr

####prueba######intento de curva roc, posta posta

####probar predict_proba(X)

fpr, tpr, thresholds = roc_curve((y_test), largo_pico_pred)

plt.plot(tpr , fpr , '-b')

plt.show()

#####idem con preddict_proba ==== largo_pico_proba

fpr, tpr, thresholds = roc_curve((y_test), largo_pico_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()


fpr, tpr, thresholds = roc_curve((y_test), profundidad_pico_pred)

plt.plot(tpr , fpr , '-b')

plt.show()

fpr, tpr, thresholds = roc_curve((y_test), profundidad_pico_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()


fpr, tpr, thresholds = roc_curve((y_test), ratio_pico_pred)

plt.plot(tpr , fpr , '-b')

plt.show()

fpr, tpr, thresholds = roc_curve((y_test), ratio_pico_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

fpr, tpr, thresholds = roc_curve((y_test), largo_aleta_pred)

plt.plot(tpr , fpr , '-b')

plt.show()

fpr, tpr, thresholds = roc_curve((y_test), largo_aleta_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

fpr, tpr, thresholds = roc_curve((y_test), masa_pred)

plt.plot(tpr , fpr , '-b')

plt.show()

fpr, tpr, thresholds = roc_curve((y_test), masa_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()
##################   
#puedo usar mas de una variable para la regresio??

##si!!!!

#para el softmax, voy a usar los features  , 'culmen_depth_mm' , 'culmen_length_mm',
#softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
#softmax_reg.fit(X, y)

X_soft_train = X_train.loc[: , ['culmen_depth_mm' , 'culmen_length_mm']]

X_soft_test = X_test.loc[: , ['culmen_depth_mm' , 'culmen_length_mm']]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=1)

softmax_reg.fit((X_soft_train) , np.asarray(y_train).ravel())

probas_C1 = softmax_reg.predict_proba((X_soft_test))

softmax_reg.score(X_soft_test , y_test)


fpr, tpr, thresholds = roc_curve((y_test), probas_C1[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

#no mejora demasiado, veamos otros hiperparaametros

softmax_reg_c10 = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)

softmax_reg_c10.fit(X_soft_train, np.asarray(y_train).ravel())

probas_C10 = softmax_reg_c10.predict_proba((X_soft_test))

fpr, tpr, thresholds = roc_curve((y_test), probas_C10[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()


softmax_reg_c100 = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=100)

softmax_reg_c100.fit(X_soft_train, np.asarray(y_train).ravel())

probas_C100 = softmax_reg_c100.predict_proba((X_soft_test))

fpr, tpr, thresholds = roc_curve((y_test), probas_C100[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()


softmax_reg_c1000 = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=1000)

softmax_reg_c1000.fit(X_soft_train, np.asarray(y_train).ravel())

probas_C1000 = softmax_reg_c1000.predict_proba((X_soft_test))

fpr, tpr, thresholds = roc_curve((y_test), probas_C1000[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

softmax_reg_c01 = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=0.1)

softmax_reg_c01.fit(X_soft_train, np.asarray(y_train).ravel())

probas_C01 = softmax_reg_c01.predict_proba((X_soft_test))

fpr, tpr, thresholds = roc_curve((y_test), probas_C01[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()


softmax_reg_c001 = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=0.01)

softmax_reg_c001.fit(X_soft_train, np.asarray(y_train).ravel())

probas_C001 = softmax_reg_c001.predict_proba((X_soft_test))

fpr, tpr, thresholds = roc_curve((y_test), probas_C001[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()



softmax_reg_c0001 = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=0.001)

softmax_reg_c0001.fit(X_soft_train, np.asarray(y_train).ravel())

probas_C0001 = softmax_reg_c0001.predict_proba((X_soft_test))

fpr, tpr, thresholds = roc_curve((y_test), probas_C0001[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

print(softmax_reg_c0001.score(X_soft_test , y_test) , softmax_reg_c001.score(X_soft_test , y_test),
softmax_reg_c01.score(X_soft_test , y_test) , softmax_reg_c1000.score(X_soft_test , y_test),
softmax_reg_c100.score(X_soft_test , y_test) , softmax_reg_c10.score(X_soft_test , y_test))

lista_softmax = [softmax_reg_c0001.score(X_soft_test , y_test) ,
softmax_reg_c001.score(X_soft_test , y_test),
softmax_reg_c01.score(X_soft_test , y_test) , softmax_reg.score(X_soft_test , y_test) ,
softmax_reg_c10.score(X_soft_test , y_test) , 
softmax_reg_c100.score(X_soft_test , y_test),softmax_reg_c1000.score(X_soft_test , y_test)]

lista_C = [0.001 , 0.01 , 0.1 , 1 , 10 , 100 , 1000]

sns.barplot(x=lista_C , y=lista_softmax)

######arranquemos por SVM

###mismo objetivo, sexar pinguinos dada una especie

#Armo una funcion que codifique sexo numericamente,  o n signos distintos

def codificar(dataframe , strcol):
    
    dataframe[strcol] = dataframe[strcol].map({'FEMALE' : 1 , 'MALE' : -1})
    
    return(dataframe)

codificar(chin , 'sex')

chin['sex'].unique()

#ahora hago el splittintg de X_train X_test , y_train , t _test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    chin.loc[: , ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g']],
    chin.loc[: , ['sex']] , test_size=0.2, random_state=512)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

##

#para ello voy a evaluar usar svm con varias variables, por separado, evaluar si svm funciona mejor 
#con alguna de ellas

#este intento fue con todas las variables

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01 , 0.1, 1, 10, 100, 1000],
              'gamma': [100,10,1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

grid.fit(np.asarray(X_train).reshape(-1 , 4) , np.asarray(y_train).ravel())

# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)
 
# print classification report
print(classification_report(y_test, grid_predictions))

print(confusion_matrix(np.asarray(y_test).ravel(), grid_predictions))

#con esto se ve que lo mejor fue SVC(C=1, gamma=0.0001) y que tiene m'as precisi'on en determinar hembras
#que machos

##

#este intento va con culmen_length

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01 , 0.1, 1, 10, 100, 1000],
              'gamma': [100,10,1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

grid.fit(np.asarray(X_train['culmen_length_mm']).reshape(-1 , 1) , np.asarray(y_train).ravel())

# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(np.asarray(X_test['culmen_length_mm']).reshape(-1,1))
 
# print classification report
print(classification_report(y_test, grid_predictions))

print(confusion_matrix(np.asarray(y_test).ravel(), grid_predictions))

#este intento va con culmen_depth

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01 , 0.1, 1, 10, 100, 1000],
              'gamma': [100,10,1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

grid.fit(np.asarray(X_train['culmen_depth_mm']).reshape(-1 , 1) , np.asarray(y_train).ravel())

# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(np.asarray(X_test['culmen_depth_mm']).reshape(-1,1))
 
# print classification report
print(classification_report(y_test, grid_predictions))

print(confusion_matrix(np.asarray(y_test).ravel(), grid_predictions))

##
#este intento va con culmen_depth, culmen_length

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01 , 0.1, 1, 10, 100, 1000],
              'gamma': [100,10,1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

grid.fit(np.asarray(X_train.loc[: , ['culmen_depth_mm','culmen_depth_mm']]).reshape(-1 , 2) , np.asarray(y_train).ravel())

# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(np.asarray(X_test.loc[: , ['culmen_depth_mm','culmen_depth_mm']]).reshape(-1,2))
 
# print classification report
print(classification_report(y_test, grid_predictions))

print(confusion_matrix(np.asarray(y_test).ravel(), grid_predictions))

#Sale ahora KNN

#la idea es clasificar tal vez viendo de a una variable a la vez, y luego
#varias juntas

from sklearn.neighbors import KNeighborsClassifier

#seguimos jugando con X_train, X_test, y_train, y_test

knn_largo_pico = KNeighborsClassifier()

knn_largo_pico.fit(X_train.loc[:,['culmen_length_mm']] , np.array(y_train).reshape(-1,1).ravel())

largo_pico_predict = knn_largo_pico.predict(np.array(X_test['culmen_length_mm']).reshape(-1,1))

largo_pico_predict_proba = knn_largo_pico.predict_proba(np.array(X_test['culmen_length_mm']).reshape(-1,1))

fpr, tpr, thresholds = roc_curve((y_test), largo_pico_predict_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

knn_profundidad_pico = KNeighborsClassifier()

knn_profundidad_pico.fit(X_train.loc[:,['culmen_depth_mm']] , np.array(y_train).reshape(-1,1).ravel())

profundidad_pico_predict = knn_profundidad_pico.predict(np.array(X_test['culmen_depth_mm']).reshape(-1,1))

profundidad_pico_predict_proba = knn_profundidad_pico.predict_proba(np.array(X_test['culmen_depth_mm']).reshape(-1,1))

fpr, tpr, thresholds = roc_curve((y_test), profundidad_pico_predict_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

knn_largo_aleta = KNeighborsClassifier()

knn_largo_aleta.fit(X_train.loc[:,['flipper_length_mm']] , np.array(y_train).reshape(-1,1).ravel())

largo_aleta_predict = knn_largo_aleta.predict(np.array(X_test['flipper_length_mm']).reshape(-1,1))

largo_aleta_predict_proba = knn_largo_aleta.predict_proba(np.array(X_test['flipper_length_mm']).reshape(-1,1))

fpr, tpr, thresholds = roc_curve((y_test), largo_aleta_predict_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

knn_masa = KNeighborsClassifier()

knn_masa.fit(X_train.loc[:,['body_mass_g']] , np.array(y_train).reshape(-1,1).ravel())

masa_predict = knn_masa.predict(np.array(X_test['body_mass_g']).reshape(-1,1))

masa_predict_proba = knn_masa.predict_proba(np.array(X_test['body_mass_g']).reshape(-1,1))

fpr, tpr, thresholds = roc_curve((y_test), masa_predict_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

knn_picos = KNeighborsClassifier()

knn_picos.fit(X_train.loc[:,['culmen_length_mm', 'culmen_depth_mm']] , np.array(y_train).reshape(-1,1).ravel())

fpr, tpr, thresholds = roc_curve((y_test), picos_predict_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

####se sigue ocn arboles de desicion...

from sklearn.tree import DecisionTreeClassifier

#usar X_train, X_test, y_train, y_test

arbol = DecisionTreeClassifier()

arbol.fit(X_train , y_train)

arbol_pred = arbol.predict(X_test)

arbol_pred_proba = arbol.predict_proba(X_test)

arbol.feature_importances_
#me  a a interesar ver predict, predict_proba, feature_importances_, y
# decision_path

#####rtuva roc del arbol

fpr, tpr, thresholds = roc_curve((y_test), arbol_pred_proba[:,0])

plt.plot(tpr , fpr , '-b')

plt.show()

from sklearn.tree import plot_tree

plot_tree(arbol)

####comparacion entre modelos? cual fue mejor??

#hagan curcas roc comparativas entre el mejor modelo de cada metodo

#pueden hacer matrices de confusion para el mejor modelo de cada metodo

#pueden probar graficar el score del mejor modelo de cada metodo

#prueben sklearn.metrics.precision_recall_curve