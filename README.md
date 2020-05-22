# KNN_Algorithm
## Hausarbeit über KNN

Hier werden mithilfe des k-nearest-neighbour Algorithmus Pilze klassifiziert. Es handelt sich um eine binäre Klassifizerung in "Giftig" und "Essbar".

Die für die Entscheidung berücksichtigten Features sind äußerliche Merkmale des Pilzes, wie z.B. Hutform, Hutoberfläche, Hutfarbe, Flecken, Geruch, Pilzlamellen, Stielform, etc...

Alle Daten liegen in Textform vor müssen daher für den Algorithmus in Zahlen formatiert werden. Dies erfolgt mithilfe des LabelEncoders.

Eine Normalisierung der Werte (in einen Zahlenraum zwischen 0 und 1) ist nicht notwendig, da der LabelEncoder immer nur ganzzahlen aufsteigend für jede Gruppe verwendet.