# KNN_Algorithm
## Hausarbeit über KNN
Der KNN-Algorithmus zählt zu den einfachsten Machine Learning Algorithmen, da er besonders durch die Bibilothek sklearn einfach zu implementieren ist. Um auf die Klassifizierung eines Datenpunktes zu schließen verwendet der KNN-Algorithmus die Annotation der nächsten Nachbarn. Folgende Parameter müssen beim Training bestimmt werden:
- n_neighbors (Anzahl der betrachteten Nachbarn)
- metric (Abstandsmaß)

Der Algorithmus zählt zu den Lazy-Learning Algorithmen, das bedeutet er speichert alle gelernte Insatzen und ruft sie bei der Klassifizierung erneut auf. Zu viele Trainingsdaten führen zu einem sehr rechenintensiven Prozess. Des weiter ist eine Vorbearbeitung der Daten nötig. Folgende Sachem müssen gemacht werden um Fehler zu vermeiden:
- Categorical Encoding (alle Datentypen in Zahlen umwandeln)
- Feature Scaling (gleiche Skalen für alle Daten, um eine stärkere Gewichtung einzelner Merkmale auszuschließen)
- Merkmalsauswahl (zu viele Dimensionen können Fehler veursachen)


Im dem vorliegenden Skript werden mithilfe des k-nearest-neighbour Algorithmus Pilze klassifiziert. Es handelt sich um eine binäre Klassifizerung in "Giftig" und "Essbar".

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