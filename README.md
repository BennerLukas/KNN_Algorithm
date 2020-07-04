# KNN_Algorithm
## Hausarbeit über KNN

Hier werden mithilfe des k-nearest-neighbour Algorithmus Pilze klassifiziert. Es handelt sich um eine binäre Klassifizerung in "Giftig" und "Essbar".

Die für die Entscheidung berücksichtigten Features sind äußerliche Merkmale des Pilzes, wie z.B. Hutform, Hutoberfläche, Hutfarbe, Flecken, Geruch, Pilzlamellen, Stielform, etc...

Alle Daten liegen in Textform vor müssen daher für den Algorithmus in Zahlen formatiert werden. Dies erfolgt mithilfe des LabelEncoders.

Bei Fehlern bitte darauf achten, dass der Dateipfad für die einzulesende CSV-Datei richtig ist.

## Datenauswahl:
Folgende beiden Features werden verwendet.

### Geruch (odor):
- Almond
- Anise
- Creoste
- Fishy
- Foul
- Musty
- None
- Pugent
- spicy

### Kappenform (cap-shape):
- bell
- conical
- convex
- flat
- knobbed
- sunken