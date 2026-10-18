# Final Talk — Sprechskript

Sprechtext auf Deutsch, Folie für Folie, passend zu den manim-Folien in `slides/talk.py`
(die Folien selbst sind englisch). Jeder **[Klick]** entspricht einem `next_slide()`,
also einem Weiterschalten im Presenter. Die Zeiten sind Richtwerte für 20 Minuten bei
ruhigem Sprechtempo (~130 Wörter pro Minute).

Alle Genauigkeitswerte sind mAP@[.5:.95] auf dem **realen** Testset, außer es steht
ausdrücklich etwas anderes dabei. Quellen: `slides/data.py` und Kapitel 4 des Manuskripts.

## Zeitplan

| Akt | Szene | Folien | Zeit | kumuliert |
|-----|-------|--------|------|-----------|
| 0 Einstieg | `Intro` | 1–4 | 1:30 | 1:30 |
| 1 Problem & Plan | `Problem` | 5–7 | 2:00 | 3:30 |
| 2 Daten | `Data` | 8–12 | 3:00 | 6:30 |
| 3 Synthetische Daten & Generatoren | `Generators` | 13–21 | 5:00 | 11:30 |
| 4 Bänder | `Bands` | 22–26 | 3:30 | 15:00 |
| 5 Modelle | `Models` | 27–30 | 3:30 | 18:30 |
| 6 Abschluss | `Close` | 31–33 | 1:30 | 20:00 |

**Wenn die Zeit knapp wird:** Folie 20 (Seeds) und Folie 25 (Punktwolke pro Klasse) nur
mit einem Satz kommentieren und durchklicken. Folie 26 (Testset) und Folie 28 (Modelle)
nicht kürzen, die tragen die Kernaussagen.

---

## Akt 0 — Einstieg (`Intro`)

### Folie 1 — Titel · 0:15

> Guten Tag zusammen. In meiner Masterarbeit ging es um Objekterkennung in Echtzeit mit
> neuronalen Netzen auf eingebetteten Plattformen. Ich erzähle das heute nicht Kapitel für
> Kapitel, sondern so, wie die Arbeit tatsächlich verlaufen ist, mit den Ideen, den
> Problemen und den Stellen, an denen mein Plan nicht aufgegangen ist.

**[Klick]**

### Folie 2 — The product · 0:20

> Angefangen hat alles mit diesem Produkt: dem AX Visio von Swarovski Optik, einem
> Fernglas mit eingebauter künstlicher Intelligenz.

**[Klick]** *(Preis-Pill erscheint)*

> Preis: viertausendsechshundert Euro. Wer so viel für ein Fernglas ausgibt, erwartet,
> dass die Erkennung funktioniert, und zwar draußen im Feld.

**[Klick]**

### Folie 3 — BirdID · 0:30

> Und das tut sie bei Vögeln schon. Im Fernglas sitzt ein Qualcomm QCS605, ein mobiler
> Chip, wie man ihn aus Smartphones von vor einigen Jahren kennt. Man richtet das Glas auf
> einen Vogel, die Kamera nimmt das Bild auf, ein neuronales Netz läuft direkt auf dem
> Gerät, und im Okular steht: Rauchschwalbe. Ohne Internet, denn im Wald oder in der
> Savanne gibt es oft keins.

**[Klick]**

### Folie 4 — Turn the dial … · 0:25

> Auf dem Wahlrad gibt es aber auch ein Symbol für Säugetiere.

**[Klick]** *(„MammalID?“ erscheint)*

> Und genau hier setzt die Arbeit an: Wie bekommt man eine zuverlässige
> Säugetiererkennung auf so ein Gerät?

**[Klick]**

---

## Akt 1 — Die naheliegende Lösung und der Plan (`Problem`)

### Folie 5 — Two models, one answer · 0:45

> Für Säugetiere gibt es bereits eine Lösung, die gut funktioniert. Hier ein Bild aus
> einer Kamerafalle.

**[Klick]** *(Box, Crop, Klassifikation laufen nacheinander ab)*

> Zuerst findet MegaDetector das Tier und zeichnet eine Bounding Box, hier mit 94 Prozent
> Konfidenz. Dieser Ausschnitt geht dann an SpeciesNet, einen Klassifikator von Google,
> und der sagt: Ozelot, mit 98,8 Prozent. Zwei Modelle, eine Antwort. Die Antwort ist
> richtig.

**[Klick]**

### Folie 5 (Fortsetzung) — Größe und Stoppuhr · 0:20

> Das Problem: Zusammen haben die beiden Modelle rund 193 Millionen Parameter. Und damit
> stellt sich eine Frage, die ich mir für später aufhebe: Wie lange braucht dieses
> Ensemble auf einem Fernglas für ein einziges Bild?

**[Klick]**

### Folie 6 — The plan · 0:40

> Mein ursprünglicher Plan sah so aus: Wir haben viele Fotos ohne Labels. Das große
> Ensemble kann sie verarbeiten, ist aber zu groß fürs Gerät. Es ist der Lehrer.

**[Klick]**

> Daneben ein kleines Modell, ein Detektor mit nur wenigen Millionen Parametern, der auf
> dem Gerät laufen könnte. Der Schüler.

**[Klick]**

> Und dazwischen Knowledge Distillation: Der Schüler lernt nicht nur aus den Labels,
> sondern aus der Wahrscheinlichkeitsverteilung des Lehrers. Das war die Idee, um die
> herum ich die Arbeit ursprünglich aufgebaut habe.

**[Klick]**

### Folie 7 — Four questions · 0:15

> Daraus sind vier Forschungsfragen entstanden: Hilft der Lehrer dem Schüler wirklich?
> Welcher Bildgenerator liefert brauchbare Trainingsdaten? Wie viel echte und wie viel
> künstliche Daten braucht ein Modell? Und: Ist das Ganze schnell genug? Am Ende komme ich
> auf alle vier zurück.

**[Klick]**

---

## Akt 2 — Daten, die es nicht gibt (`Data`)

### Folie 8 — Which animals? · 0:30

> Bevor man etwas trainieren kann, braucht man Daten. Und zuerst muss man festlegen, was
> überhaupt erkannt werden soll. SpeciesNet kennt rund zweieinhalbtausend Arten.

**[Klick]**

> Für das Fernglas braucht man aber nur Säugetiere, und nur solche, die man draußen
> tatsächlich durch ein Fernglas sieht. Keine Vögel, keine Reptilien, keine Arten, die
> nur nachts unter der Erde leben.

**[Klick]**

### Folie 9 — Hedgehogs · 0:25

> Und dann stellt sich die Frage, wie fein man unterscheiden soll. Das hier sind Igel. 

**[Klick]** *(„17 species → 1 class“)*

> Siebzehn verschiedene Arten. Auf einem Foto aus fünfzig Metern Entfernung kann die
> niemand auseinanderhalten, auch kein Modell. Also wird daraus eine Klasse:
> „Igel“. Manche Klassen sind deshalb Arten, andere Gattungen oder ganze Familien.

**[Klick]**

### Folie 10 — 225 classes · 0:15

> Am Ende waren es 225 Klassen, die meisten davon Raubtiere und Paarhufer. Der
> Product Owner bei Swarovski hat die Auswahl abgenommen.

**[Klick]**

### Folie 11 — Images per class · 0:45

> Dann die Frage: Wie viele Bilder gibt es pro Klasse? Startpunkt waren Fotos aus GBIF,
> einer großen Biodiversitätsdatenbank. Jeder Balken ist eine Klasse, die Achse ist
> logarithmisch. 114 Klassen haben weniger als hundert Bilder, und 30 Klassen kein
> einziges.

**[Klick]** *(weitere Quellen)*

> Also habe ich vier weitere Quellen dazugenommen: iNaturalist, Wikimedia, OpenImages und
> images.cv. Das hilft vielen Klassen deutlich, aber 29 Klassen bleiben trotzdem unter
> hundert Bildern. Für ein sauberes Training reicht das nicht.

**[Klick]**

### Folie 12 — Is this an animal? · 0:45

> Dazu kommt: Nicht alles, was im Internet als Tier markiert ist, ist auch ein Tier. Das
> hier sind echte Bilder aus Wikimedia, unter den Kategorien Gepard, Schimpanse und Tiger:
> eine ägyptische Wandmalerei, ein Affe an der Schreibmaschine, ein Uni-Maskottchen, eine
> Verbreitungskarte.

**[Klick]** *(Filterpipeline)*

> Deshalb läuft jedes Bild durch eine Filterkette: zuerst billige Prüfungen auf Metadaten
> und Bildqualität, dann MegaDetector, dann ein Vision-Language-Modell als zweite Meinung,
> und am Ende SpeciesNet. Die unklaren Fälle, knapp achtzehntausend Bilder, habe ich mir
> von Hand angesehen.

**[Klick]** *(SpeciesNet hervorgehoben)*

> Ein Detail, das später wichtig wird: Die Artenlabels kommen am Ende von SpeciesNet,
> also genau von dem Modell, das im Plan der Lehrer sein sollte.

**[Klick]**

---

## Akt 3 — Fehlende Daten erzeugen und welchem Generator man trauen kann (`Generators`)

### Folie 13 — Binturong · 0:40

> Was macht man mit Klassen, für die es kaum Fotos gibt? Man erzeugt sie. Das hier ist
> ein Binturong, ein Marderbär. Links echte Fotos aus Wikimedia.

**[Klick]** *(vier Generatoren)*

> Daneben dieselbe Art aus vier Bildgeneratoren: Gemini, RealVisXL, Stable Diffusion 3.5
> und FLUX. Man sieht sofort: Einige davon haben eher einen Bären gemalt als einen
> Binturong.

**[Klick]** *(„looked best“)*

> Ich habe mich damals für Gemini entschieden, weil die Bilder am realistischsten aussahen.
> Ob das ein guter Grund war, dazu gleich mehr.

**[Klick]**

### Folie 14 — Three zebras · 0:20

> Wie schwer die Aufgabe ist, sieht man an Zebras. Es gibt drei Arten: Grévyzebra,
> Bergzebra und Steppenzebra. Der Unterschied liegt in den Streifen, also in Details. Ein
> Generator muss genau diese Details richtig treffen, sonst lernt das Modell das Falsche.

**[Klick]**

### Folie 15 — Der Prompt · 0:30

> Deshalb sind die Prompts sehr lang. Das hier ist ein einziger Prompt für ein einziges
> Bild: die Artbeschreibung aus Wikipedia, Lebensraum, Verhalten, und dazu eine Szene mit
> Pose, Umgebung, Licht und Entfernung.

**[Klick]** *(Schlüsselmerkmale)*

> Dazu die Merkmale, an denen man die Art erkennt: beim Grévyzebra die schmalen
> Streifen, der weiße Bauch, die großen runden Ohren. Insgesamt etwa zweitausendsechshundert
> Tokens pro Bild.

**[Klick]**

### Folie 16 — 23,850 synthetic images · 0:15

> Am Ende waren es 23.850 synthetische Bilder, zum Listenpreis rund 785 Dollar. An dieser
> Stelle danke an das CTO-Office von inovex für die Unterstützung.

**[Klick]**

### Folie 17 — Same prompt, eight sources · 0:35

> Zurück zur Frage: War Gemini die richtige Wahl? Das habe ich nachträglich in einem
> kontrollierten Experiment geprüft. Hier ein Löwe, derselbe Prompt, ein echtes Foto und
> sieben Generatoren.

**[Klick]** *(Versuchsaufbau)*

> Der Aufbau: zwölf Konfigurationen, jeweils 1.200 Bilder für dieselben zwölf Klassen,
> derselbe Detektor, dasselbe Training, und bewertet wird auf 9.742 echten Fotos. Gezählt
> hat also nicht, wie schön die Bilder aussehen, sondern wie gut ein Detektor danach echte
> Tiere erkennt.

**[Klick]**

### Folie 18 — Which generator helps the detector? · 0:45

> Das Ergebnis: Blau sind die kommerziellen API-Modelle, grau die lokal laufenden Modelle.
> Alle API-Modelle liegen vor allen lokalen Modellen.

**[Klick]** *(„my pick: #3“)*

> Meine Wahl, Gemini, landet auf Platz drei. Nicht schlecht, aber auch nicht vorne. Und
> man sieht am langen Fehlerbalken, wie stark das Ergebnis zwischen Trainingsläufen
> schwankt.

**[Klick]** *(„#1 = cheapest setting“)*

> Und das Überraschende: Platz eins ist gpt-image-2 in der günstigsten Qualitätsstufe.
> Die mittlere Stufe kostet fast dreimal so viel und ist nicht besser. Bei den lokalen
> Modellen ist es genauso: Das teuerste, HiDream, mit fast 48 GPU-Stunden, landet auf dem
> vorletzten Platz. Weder der optische Eindruck noch der Preis sagen voraus, welche Bilder
> beim Training helfen.

**[Klick]**

### Folie 19 — Shorter prompt, worse data · 0:30

> Was es stattdessen voraussagt, zeigt dieser Versuch: dasselbe Modell, einmal mit dem
> vollen Prompt, einmal mit einer Kurzfassung von etwa 55 Tokens.

**[Klick]** *(Kurzprompt-Balken, −45 % / −48 %)*

> Mit dem kurzen Prompt verliert der Detektor fast die Hälfte seiner Genauigkeit, bei
> beiden Modellen. Das ist mehr als der Abstand zwischen den meisten Generatoren. Wie
> detailliert man beschreibt, ist also wichtiger als die Wahl des Generators.

**[Klick]**

### Folie 20 — One run can lie · 0:30

> Eine Lektion am Rande. Nach dem ersten Trainingslauf lag meine Wahl bei 0,064, deutlich
> hinter ihrer günstigeren Schwester.

**[Klick]**

> Nach dem zweiten Lauf: praktisch gleichauf.

**[Klick]**

> Nach dem dritten: vorne. Ein Unterschied in dieser Größenordnung aus einem einzigen Lauf
> sagt also nichts aus. Das muss man im Kopf behalten, denn bei den großen Modellen später
> konnte ich aus Zeitgründen nur einen Lauf pro Modell machen.

**[Klick]**

### Folie 21 — Rare animals are a blind spot · 0:45

> Und jetzt der unangenehme Teil. Das hier ist die Genauigkeit pro Klasse und pro
> Generator. Je heller, desto besser.

**[Klick]** *(seltene Klassen markiert)*

> Die oberen sechs Zeilen sind die seltenen Arten, also genau die, für die ich die
> synthetischen Bilder überhaupt gebraucht habe. Dort liegt jeder Generator unter 0,07.
> Für diese Klassen kann das Experiment die Generatoren nicht unterscheiden.

**[Klick]** *(Saiga)*

> Ein Beispiel, warum: Das ist eine Saiga-Antilope mit ihrer typischen rüsselartigen Nase.
> Die lokalen Generatoren malen stattdessen eine gewöhnliche Gazelle. Ein Generator kann
> offenbar nur zeichnen, was er schon oft gesehen hat, und seltene Tiere hat er eben
> selten gesehen.

**[Klick]**

---

## Akt 4 — Bänder: Was können synthetische Bilder leisten? (`Bands`)

### Folie 22 — Same budget, different mix · 0:35

> Um zu messen, wie viel synthetische Bilder bringen, habe ich die Klassen nach ihrer
> Datenlage in vier Bänder eingeteilt. Jedes Quadrat steht für zehn Bilder, grün
> umrandet heißt synthetisch.
> Band A, die 50 Klassen mit den wenigsten Fotos: nur synthetische Bilder.
> Band B: halb echt, halb synthetisch. Band C: nur echte Fotos. Und Band D, die gut
> dokumentierten Arten, mit bis zu 1.500 echten Bildern als Referenz.

**[Klick]** *(„200 images each“)*

> Wichtig: A, B und C bekommen genau gleich viele Bilder, je 200. Nur die Mischung ist
> anders. Damit kann man echte und künstliche Bilder bei gleichem Budget vergleichen.
> Einschränkung: In jedem Band stecken andere Arten, der Vergleich läuft also über
> verschiedene Tiere hinweg.

**[Klick]**

### Folie 23 — Die Null · 0:40

> Das erste Ergebnis für Band A sah so aus: null. Nicht ungefähr null, sondern exakt null.

**[Klick]** *(Bug)*

> Ein Ergebnis von exakt null ist fast nie ein Ergebnis, sondern fast immer ein Fehler.
> Und so war es auch: Der Data Loader hat nur die Annotationen der echten Bilder gelesen.
> Die synthetischen Bilder hatte das Modell nie gesehen.

**[Klick]** *(Zähler auf 0,300)*

> Nach der Korrektur und dem erneuten Training: 0,30. Damit können wir jetzt die
> eigentliche Frage beantworten.

**[Klick]**

### Folie 24 — Half synthetic is fine. All synthetic is not. · 0:40

> Hier die Genauigkeit des kleinen Detektors pro Band, gemessen auf echten Fotos. Band B,
> halb synthetisch, kommt auf 0,59. Band C, nur echte Fotos, ebenfalls 0,59.

**[Klick]** *(„=“ und „½“)*

> Das heißt: Die Hälfte eines kleinen echten Datensatzes lässt sich durch synthetische
> Bilder ersetzen, ohne messbaren Verlust. Beim großen Ensemble ist es genauso, nur beim
> älteren YOLOv5s verliert man etwas.
> Aber Band A, nur synthetisch, erreicht nur etwa die Hälfte. Synthetische Daten strecken
> also einen echten Datensatz, sie ersetzen ihn nicht.

**[Klick]**

### Folie 25 — Not every animal is the same · 0:30

> Hinter den Durchschnitten steckt eine große Streuung. Jeder Punkt ist eine Klasse, der
> weiße Strich ist der Median.

**[Klick]** *(Hyäne und Mangaben)*

> In Band A, also rein synthetisch trainiert, erreicht die Streifenhyäne 0,48, die
> Mangaben dagegen fast null. Meine Vermutung, ausdrücklich eine Hypothese: Arten mit
> einem klaren, einheitlichen Erscheinungsbild lassen sich gut generieren. Klassen, die
> in der Realität sehr vielfältig aussehen, etwa eine ganze Affengattung, eher nicht,
> weil der Generator immer eine idealisierte Version zeichnet.

**[Klick]**

### Folie 26 — Was the test fair? · 0:50

> Noch eine Frage zur Messung selbst. Das Testset bestand aus echten und aus
> synthetischen Bildern. Links ein synthetisches Testbild: ein Tier, mittig, freie Sicht.
> Rechts ein echtes: mehrere Zebras, Gnus, teilweise verdeckt.

**[Klick]** *(Balken)*

> Auf den synthetischen Testbildern schneiden alle Modelle viel besser ab, in Band A mit
> 0,93 gegenüber 0,20 auf echten Fotos. Und das gilt sogar für Klassen, die nie ein
> synthetisches Trainingsbild gesehen haben. Ein Teil davon ist einfach Schwierigkeit:
> Das fertige Ensemble, das nie mit synthetischen Bildern trainiert wurde, ist dort
> ebenfalls um etwa 0,2 besser.

**[Klick]** *(„from here on: real photos only“)*

> Die Konsequenz: Ab hier zeige ich nur noch Werte auf echten Fotos, denn nur die sagen
> etwas über den Einsatz im Feld aus.

**[Klick]**

---

## Akt 5 — Welches Modell? (`Models`)

### Folie 27 — Three candidates · 0:30

> Dann zu den Modellen. Drei Kandidaten, die Fläche der Kreise entspricht der Zahl der
> Parameter: das Ensemble aus MegaDetector und SpeciesNet mit 193 Millionen, YOLOv5s mit
> 7,6 Millionen und YOLO26n mit 2,7 Millionen.

**[Klick]** *(„in the binocular today“)*

> YOLOv5s ist übrigens die Architektur, die heute schon im Fernglas für die Detektion
> läuft.

**[Klick]**

### Folie 28 — Accuracy on real photos · 1:00

> Die Genauigkeit auf echten Fotos: Das feinjustierte Ensemble liegt mit 0,60 vorne. Der
> Stern bedeutet, dass das eine Obergrenze ist, denn MegaDetector hat auch die
> Ground-Truth-Boxen erzeugt, gegen die es bewertet wird. Das kleine YOLO26n kommt auf
> 0,53 und schlägt damit sowohl das Ensemble ohne Fine-Tuning als auch das größere
> YOLOv5s.

**[Klick]** *(Größe und Zeit pro Bild)*

> Jetzt die Frage von vorhin mit der Stoppuhr. Gemessen auf einem Raspberry Pi 400 als
> Stellvertreter für den Fernglas-Chip, nur auf der CPU, in voller Genauigkeit, ohne
> Quantisierung. Das Ensemble braucht etwa 35 Sekunden pro Bild. Bis dahin ist jedes Tier
> längst weg. YOLOv5s braucht knapp eine Sekunde, YOLO26n 0,42 Sekunden.

**[Klick]** *(„83× faster · 71× smaller · −0.07 mAP“)*

> Zusammengefasst: 83-mal schneller und 71-mal kleiner als das Ensemble, bei nur 0,07
> weniger Genauigkeit. Echtzeit ist das noch nicht, dafür fehlt der nächste Schritt,
> also Quantisierung und der Beschleuniger des Chips. Aber wir sind von einer halben
> Minute pro Bild bei unter einer halben Sekunde angekommen.
> Eine Randbemerkung: YOLO26n hat ohne jedes Tuning das ältere YOLOv5s in allem
> geschlagen. Zwei Modellgenerationen Fortschritt bringen offenbar mehr, als ich durch
> Feintuning des alten Modells hätte herausholen können.

**[Klick]**

### Folie 29 — Finding is easy. Naming is hard. · 0:40

> Wo verliert das kleine Modell seine Genauigkeit? Wenn man nur fragt, ob es das Tier
> findet, egal welche Art, liegt es bei 0,78. Mit der richtigen Art sind es 0,53.

**[Klick]** *(Elefanten)*

> Das Finden ist also nicht das Problem, sondern das Benennen. Das deutlichste Beispiel:
> Asiatische Elefanten werden 1.545-mal als Afrikanische erkannt, umgekehrt nur neunmal.
> Das Modell verwechselt die beiden nicht einfach, es fasst die eine Art in die andere
> zusammen. An der Größe des Testsets liegt es nicht, es gibt sogar mehr Testbilder von
> Asiatischen Elefanten.

**[Klick]**

### Folie 30 — Did the teacher help? · 0:50

> Und jetzt zurück zum ursprünglichen Plan, der Distillation. In grau der Schüler, der vom
> Lehrer gelernt hat.

**[Klick]** *(direktes Training)*

> In grün derselbe Schüler, einfach direkt auf den Daten trainiert.

**[Klick]** *(„direct training wins in every band“)*

> Das direkte Training gewinnt in jedem Band, und am deutlichsten ausgerechnet in Band A,
> wo ich mir vom Lehrer am meisten erhofft hatte.
> Warum? Erinnern wir uns: Die Labels kamen von SpeciesNet, also vom Lehrer selbst. Seine
> weichen Verteilungen wiederholen im Wesentlichen die harten Labels und weichen nur dort
> ab, wo er sich irrt. Und eine Art hat genau eine richtige Antwort, da gibt es wenig
> „dunkles Wissen“ weiterzugeben.
> Einschränkend: Getestet habe ich nur eine Konfiguration, und die beiden Läufe hatten
> unterschiedliche Batch-Größen. Distillation ist damit nicht grundsätzlich widerlegt,
> aber in dieser Form hat sie nichts gebracht.

**[Klick]**

---

## Akt 6 — Abschluss (`Close`)

### Folie 31 — Four questions, four answers · 0:45

> Damit zurück zu den vier Fragen vom Anfang.

**[Klick]** *(✗)*

> Hilft der Lehrer dem Schüler? In dieser Form nicht.

**[Klick]** *(✓)*

> Welcher Generator? Wichtiger als das Modell ist der Prompt, und weder Preis noch
> optischer Eindruck sagen die Qualität voraus.

**[Klick]** *(~)*

> Echte oder synthetische Daten? Die Hälfte lässt sich ersetzen, aber nicht alles.

**[Klick]** *(~)*

> Und ist es schnell genug? Ein kleiner Detektor kommt mit einem Bruchteil der Größe sehr
> nah an das große Ensemble heran. Für Echtzeit fehlt noch die Optimierung für den Chip,
> das wäre der nächste Schritt, zusammen mit einer Rückfallebene, die bei Unsicherheit
> die Gattung statt der Art nennt.

**[Klick]**

### Folie 32 — Which one is real? · 0:30

> Zum Schluss eine kleine Frage: Welches dieser beiden Zebras ist echt?

*(kurz warten)*

**[Klick]**

> Das linke ist generiert, das rechte ist ein echtes Foto. Vor ein paar Jahren konnte kein
> Generator Streifenmuster in dieser Qualität erzeugen. Die Modelle werden so schnell
> besser, dass meine Zahlen in ein, zwei Jahren wahrscheinlich überholt sind. Deshalb ist
> der Versuchsaufbau so angelegt, dass man ihn mit neuen Generatoren einfach noch einmal
> laufen lassen kann.

**[Klick]**

### Folie 33 — Thank you · 0:10

> Vielen Dank für die Aufmerksamkeit. Ich freue mich auf Ihre Fragen.

---

## Vorbereitung auf Fragen

Kurze Antworten auf naheliegende Nachfragen, mit den Zahlen dazu.

| Frage | Antwort |
|-------|---------|
| Warum Raspberry Pi statt QCS605? | Der Hexagon-DSP des QCS605 hängt an einer Toolchain von 2018 mit schlecht dokumentierter Operator-Unterstützung. Der Pi 400 ist laut CPU-Vergleich etwa 10–15 % langsamer als die Kryo-CPU des QCS605, also eine konservative Schätzung: 423 ms auf dem Pi entsprechen etwa 360–380 ms auf der CPU des Ziels. Den DSP gibt es auf dem Pi nicht. |
| Wie kommt man auf Echtzeit? | Quantisierung, ideal quantisierungsbewusstes Training, dann der beschleunigte Integer-Pfad des Chips. Laut Hersteller schafft YOLO26n auf einem Raspberry Pi 5 etwa 68 ms. Gemessen habe ich das nicht. |
| Warum nur ein Lauf pro Modell? | Jeder Lauf dauerte auf einer A40 zwischen 21 Stunden und 8,8 Tagen. Konfidenzintervalle fehlen deshalb, Unterschiede unter etwa 0,03 sind nicht belastbar (siehe Seed-Folie). |
| Warum ist die Obergrenze beim Ensemble wichtig? | MegaDetector hat die Ground-Truth-Boxen erzeugt. Beim Ensemble ist mAP gleich mAP50, was kein anderer Detektor erreicht, die Lokalisierung wird also teilweise gegen die eigene Ausgabe bewertet. |
| Warum war der Mixed-Testset-Wert ursprünglich der Hauptwert? | Die 50 synthetischen Testbilder pro Klasse sollten seltene Klassen mit wenigen echten Testbildern stabilisieren. Die Analyse hat gezeigt, dass der synthetische Anteil alle Werte nach oben verzerrt. Deshalb berichte ich hier nur reale Werte. |
| Wurden die synthetischen Bilder automatisch annotiert? | Ja, die Boxen kamen von MegaDetector. Für den Generatorvergleich betrifft das alle Konfigurationen gleich, die Rangfolge bleibt also gültig, die absoluten Werte sind vorläufig. |
| Warum keine NanoDet, PicoDet usw.? | Aus Zeitgründen. Die Detektorfamilie war durch das bestehende Produkt vorgegeben, und das jüngste kleinste Modell wurde der Schüler. Ein Architekturvergleich mit Hyperparametersuche ist der naheliegende nächste Schritt. |
| Warum ist das Ensemble-Fine-Tuning in Band A schlechter? | Nach dem Fine-Tuning auf 225 Klassen sinkt Band A auf echten Fotos von 0,32 auf 0,18. Der Klassifikator verliert allgemeine Erkennungsfähigkeit bei Klassen, die er nur synthetisch gesehen hat. |
| Kosten der Bildgenerierung? | 23.850 Bilder, rund 785 $ zum Listenpreis (die Gemini-Abrechnung bis Mitte Juni zeigte 647,29 €). Für die vom Hersteller empfohlenen 1.500 Bilder pro Klasse wären es etwa vierzehnmal so viele. |
