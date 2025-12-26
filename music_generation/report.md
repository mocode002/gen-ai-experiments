# LSTM pour la Génération Musicale (Format ABC) 🎵

## 1. Introduction
Ce projet explore l'utilisation des **Réseaux de Neurones Récurrents (RNN)**, et plus spécifiquement des cellules **LSTM (Long Short-Term Memory)**, pour la génération de séquences créatives. L'objectif est d'entraîner un modèle capable de composer de nouvelles mélodies de musique traditionnelle irlandaise au format **ABC notation**.

Le défi principal réside dans la capacité du modèle à apprendre non seulement la syntaxe des caractères ASCII, mais aussi la structure musicale sous-jacente (mesures, rythme, répétitions).

## 2. Dataset et Prétraitement
Nous utilisons le dataset **"Irishman"**, composé de partitions au format JSON.

### Notation ABC
La notation ABC est un format texte compact. Exemple :
```text
X: 1
T: The Title
M: 4/4
K: G
GABc dedB | ...

```

### Pipeline de Données

1. **Extraction du vocabulaire** : Identification des caractères uniques (notes, barres de mesure, métadonnées).
2. **Mapping** : Création de deux dictionnaires `char_to_idx` et `idx_to_char` pour la vectorisation.
3. **Padding** : Uniformisation de la longueur des séquences pour permettre l'entraînement par batchs via un `DataLoader` PyTorch.

## 3. Architecture du Modèle

Le modèle est construit avec **PyTorch** et suit cette architecture :

1. **Embedding Layer** : Transforme les indices de caractères (discrets) en vecteurs denses continus, capturant des relations sémantiques entre les caractères.
2. **LSTM Layer** : Cœur du modèle. Contrairement aux RNN simples, le LSTM gère mieux les dépendances à long terme (essentiel pour la structure musicale) grâce à ses mécanismes de *gates* (oubli, entrée, sortie).
3. **Dense Layer (Fully Connected)** : Projette la sortie du LSTM vers la taille du vocabulaire pour prédire le caractère suivant.

```python
# Résumé de l'architecture
self.embedding = nn.Embedding(vocab_size, embedding_dim)
self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
self.fc = nn.Linear(hidden_size, vocab_size)
```

## 4. Entraînement

* **Loss Function** : CrossEntropyLoss (problème de classification multi-classes caractère par caractère).
* **Optimiseur** : Adam.
* **Techniques** : Utilisation de TensorBoard pour le logging et *Early Stopping* pour éviter l'overfitting.

## 5. Génération de Musique

La génération s'effectue caractère par caractère. À chaque pas de temps, la prédiction est réinjectée comme entrée pour le pas suivant.

* Nous avons exploré l'échantillonnage probabiliste plutôt que l'approche purement "Greedy" (prendre toujours la probabilité max) pour introduire de la variété et de la créativité dans les mélodies.

### Exemple de résultat généré :

```text
X:4
M:3/4
K:A
 A3 A A :: g | f2 ef | g2 fe | f2 ed | e3 e | f4 | d2 ef | g2 fe | f2 f2 | e2 e/^d/e/d/ | c4 | B3 z | d2 B3 | A2 z2 | F2 D2 | F2 c2 | A4 G2 | F2 F2 G2 :| A2 c2 | e3 e | 
 e2 a2 | e2 a2 | gf ed | c2 A2
```

## 6. Conclusion

Ce projet a permis de valider l'efficacité des LSTM pour la modélisation de séquences complexes. Le modèle parvient à respecter la syntaxe ABC et à produire des structures musicales cohérentes, bien que des incohérences harmoniques puissent persister sur de très longues séquences.