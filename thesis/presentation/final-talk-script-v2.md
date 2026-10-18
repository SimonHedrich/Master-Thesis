# Final Talk v2 — Sprechskript

Sprechtext auf Deutsch zur überarbeiteten Fassung der Folien (`slides/talk_v2.py`, Szenen
`UseCase`, `DataV2`, `Generative`, `Bands2`, `ModelsV2`, `Summary`). Die Folien selbst
sind englisch. Jeder **[Klick]** entspricht einem `next_slide()`.

Die Geschichte ist diesmal nicht chronologisch erzählt, sondern als Vorgehen, das Schritt
für Schritt aufeinander aufbaut: Anwendungsfall → Daten → generierte Bilder → Ergänzung
der Klassen → Modelle auf der Hardware → Zusammenfassung.

Die Folien zeigen bewusst wenig Text. Was dort nicht steht (Kamerafallen, Nashörner, was
hinter den drei Herausforderungen steckt, woher die Artenlabels kommen), steht hier im
Sprechtext.

Alle Genauigkeitswerte sind mAP@[.5:.95] auf **echten** Fotos, außer es steht etwas
anderes dabei. Quellen: `slides/data.py` und Kapitel 3 und 4 des Manuskripts.

## Zeitplan

| Akt | Szene | Folien | Zeit | kumuliert |
|-----|-------|--------|------|-----------|
| 1 Anwendungsfall | `UseCase` | 1–6 | 3:30 | 3:30 |
| 2 Daten | `DataV2` | 7–11 | 3:15 | 6:45 |
| 3 Generierte Bilder | `Generative` | 12–17 | 4:40 | 11:25 |
| 4 Klassen ergänzen | `Bands2` | 18–21 | 3:50 | 15:15 |
| 5 Modelle & Hardware | `ModelsV2` | 22–25 | 2:50 | 18:05 |
| 6 Zusammenfassung | `Summary` | 26–27 | 1:25 | 19:30 |

**Wenn die Zeit knapp wird:** Folie 9 (Klassenliste), die Punktwolke auf Folie 20 und
Folie 24 (Elefanten) mit je einem Satz durchklicken. Folie 21 (Testdaten) und Folie 23
(Modelle) nicht kürzen.

---

## Akt 1 — Der Anwendungsfall (`UseCase`)

### Folie 1 — Titel · 0:15

> Guten Tag zusammen. Meine Masterarbeit beschäftigt sich mit Objekterkennung in Echtzeit
> auf eingebetteten Plattformen. Ich zeige heute Schritt für Schritt, wie man von einem
> konkreten Anwendungsfall zu einem fertigen Modell kommt und was man auf dem Weg dorthin
> lernt.

**[Klick]**

### Folie 2 — Das Produkt · 0:15

> Der Anwendungsfall ist dieses Fernglas: das AX Visio von Swarovski Optik, ein Fernglas
> mit eingebauter künstlicher Intelligenz, für viertausendsechshundert Euro.

**[Klick]**

### Folie 3 — BirdID · 0:45

> Bei Vögeln funktioniert die Erkennung schon heute, und zwar so: Der Vogel draußen wird
> von der Kamera im Fernglas aufgenommen.

**[Klick]**

> Auf dem Chip im Gerät, einem Qualcomm QCS605, läuft ein neuronales Netz. Das Ergebnis
> wird nicht auf einem Bildschirm angezeigt, sondern als Overlay erzeugt.

**[Klick]**

> Und dieses Overlay wird über ein kleines Display ins Okular eingespiegelt. Man sieht
> also den Vogel ganz normal durch die Optik und darüber die Information: Rauchschwalbe.
> Alles direkt auf dem Gerät, ohne Internet.

**[Klick]**

### Folie 4 — MammalID · 0:30

> Auf dem Wahlrad gibt es auch einen Modus für Säugetiere, MammalID.

**[Klick]**

> Dahinter stehen bisher zwei frei verfügbare Modelle: MegaDetector, der Tiere auf Bildern
> findet, und SpeciesNet von Google, das sie bestimmt.

**[Klick]**

### Folie 5 — Model Ensemble · 1:15

> So arbeiten die beiden zusammen. Hier ein Bild aus einer Kamerafalle.

**[Klick]**

> MegaDetector findet das Tier und zeichnet eine Bounding Box, mit 94 Prozent Konfidenz.
> Der Ausschnitt geht an SpeciesNet, und das sagt: Ozelot, mit 98,8 Prozent.

**[Klick]**

> Zusammen sind das rund 193 Millionen Parameter. Das merken wir uns für später.
>
> Wichtig ist, wofür dieses Ensemble gebaut wurde: für Kamerafallen, also für Bilder wie
> dieses, oft nachts im Infrarot, aus fest montierten Kameras. Durch ein Fernglas sieht
> man etwas anderes: Tageslicht, große Entfernung und zum Teil andere Tiere. Ein Beispiel
> sind Nashörner. Die sind in den Trainingsdaten von SpeciesNet praktisch nicht
> vertreten. In meinen Daten hat SpeciesNet von 1.510 Nashornfotos gerade einmal 40 als
> Nashorn erkannt.
> Meine Hoffnung war deshalb: Ein Modell, das gezielt auf diesen Anwendungsfall trainiert
> wird, also auf Säugetiere, die man mit einem Fernglas beobachtet, kann für die Nutzer
> bessere Ergebnisse liefern.

**[Klick]**

### Folie 6 — Three challenges · 0:30

> Dafür gibt es drei Herausforderungen. Erstens die Daten: 225 Säugetierklassen, von
> denen viele kaum fotografiert werden. Zweitens die Hardware: ein mobiler Chip, offline.
> Und drittens die Geschwindigkeit: Die Antwort muss kommen, solange das Tier noch im Blick
> ist.
> Und damit fängt man an, wo jedes solche Projekt anfängt: bei den Daten.

**[Klick]**

---

## Akt 2 — Daten (`DataV2`)

### Folie 7 — SpeciesNet · 0:40

> Zuerst muss man festlegen, was das Modell überhaupt erkennen soll. SpeciesNet kennt rund
> zweieinhalbtausend Arten, sortiert nach der biologischen Hierarchie von Stamm bis Art.
> Darunter sind nicht nur Säugetiere, sondern auch Vögel, Reptilien, Amphibien und sogar
> Spinnen. Für das Fernglas braucht man davon nur die Säugetiere, die man draußen
> tatsächlich durch ein Fernglas sieht.

**[Klick]**

### Folie 8 — Igel · 0:25

> Die zweite Frage ist, wie fein man unterscheiden will.

**[Klick]** *(„17 species → 1 class“)*

> Das hier sind siebzehn Igelarten. Aus fünfzig Metern Entfernung kann die niemand
> auseinanderhalten. Daraus wird also eine Klasse. Deshalb sind manche Klassen Arten,
> andere Gattungen oder ganze Familien.

**[Klick]**

### Folie 9 — 225 Custom Classes · 0:20

> Am Ende stehen 225 Klassen, hier alle auf einen Blick, sortiert nach Ordnung. Die meisten
> davon sind Raubtiere und Paarhufer. Der Product Owner bei Swarovski hat diese Auswahl
> abgenommen.

**[Klick]**

### Folie 10 — Images per class · 1:00

> Als Nächstes braucht man Bilder für jede Klasse. Startpunkt waren Fotos aus GBIF, einer
> großen Biodiversitätsdatenbank. Jeder Balken ist eine Klasse, die Achse ist
> logarithmisch: 114 Klassen haben weniger als hundert Bilder, 30 gar keine.

**[Klick]** *(Quellen farbig gestapelt)*

> Also kamen vier weitere Quellen dazu, und zwar nur Bilder mit Lizenzen, die eine
> kommerzielle Nutzung erlauben. Jede Farbe ist eine Quelle, gezählt sind die Bilder, die
> die Filter überstanden haben: GBIF mit rund 32.000, iNaturalist mit Abstand am meisten,
> rund 400.000, dazu Wikimedia mit 12.500, OpenImages mit 7.700 und images.cv mit 5.500
> Bildern. Das hilft vielen Klassen. Aber 29 Klassen bleiben unter hundert Bildern, und
> für viele weitere reicht es nur knapp.

**[Klick]**

### Folie 11 — Data Quality · 0:50

> Und nicht jedes Bild, das als Tier markiert ist, zeigt auch eines. Das sind echte
> Wikimedia-Bilder aus den Kategorien Gepard, Schimpanse und Tiger: eine ägyptische
> Wandmalerei, ein Affe an der Schreibmaschine, ein Uni-Maskottchen, eine
> Verbreitungskarte.

**[Klick]** *(Filterkette)*

> Deshalb läuft jedes Bild durch eine Filterkette: Metadaten, Bildqualität, MegaDetector,
> ein Vision-Language-Modell als zweite Meinung und am Ende SpeciesNet. Die unklaren
> Fälle, knapp achtzehntausend Bilder, habe ich von Hand geprüft.
> Wichtig für später: Die Artenlabels stammen damit von SpeciesNet.
> Und es bleibt das Problem vom Anfang: Für viele Klassen gibt es einfach zu wenige
> Fotos. Also muss man sich überlegen, wie man sonst an Bilder kommt.

**[Klick]**

---

## Akt 3 — Generierte Bilder (`Generative`)

### Folie 12 — Synthetic images? – Binturong · 0:50

> Die naheliegende Idee: Bilder generieren. Bildgeneratoren sind in den letzten Jahren
> enorm besser geworden. Hier ein Binturong, ein Marderbär. Links echte Fotos.

**[Klick]** *(lokale Modelle)*

> Daneben drei Generatoren, die man lokal auf einer eigenen GPU laufen lassen kann:
> RealVisXL, Stable Diffusion 3.5 und FLUX. Man sieht: Das sind eher Bären als
> Binturongs.

**[Klick]** *(Gemini)*

> Und hier Gemini, ein kommerzielles Modell über eine API. Optisch ist der Unterschied
> deutlich. Mit Gemini habe ich deshalb den eigentlichen Datensatz erzeugt.

**[Klick]**

### Folie 13 — Drei Zebras · 0:35

> Wie genau die Bilder sein müssen, zeigen Zebras: Grévyzebra, Bergzebra, Steppenzebra,
> jeweils in einer anderen Umgebung. Der Unterschied liegt in den Streifen. Genau diese
> Details muss der Generator treffen.

**[Klick]** *(Titel „Gemini 3.1 Flash Image“)*

> Alle drei Bilder hat Gemini 3.1 Flash Image erzeugt.

**[Klick]**

### Folie 14 — Prompt · 0:50

> Damit der Generator diese Details trifft, sind die Prompts lang. Das ist der Prompt für
> einen einzigen Löwen bei Gemini: die Artbeschreibung aus Wikipedia, Lebensraum,
> Verhalten, eine Szene mit Pose, Umgebung, Licht und Entfernung und die Merkmale, an denen
> man die Art erkennt. Über viertausend Wörter, im Schnitt sind es etwa 1.300 Wörter pro
> Bild.

**[Klick]** *(immer kürzere Prompts)*

> Das Problem: Nicht jeder Generator liest so viel. FLUX.2 klein liest höchstens 512
> Tokens, Stable Diffusion 3.5 Medium 256, und RealVisXL nur 75 Tokens, das ist alles,
> was ältere Text-Encoder lesen. Man muss den Text nicht lesen können. Wichtig ist der
> Unterschied in der Länge.

**[Klick]**

### Folie 15 — Training set comparison · 1:00

> Ob die Bilder optisch gut aussehen, ist aber nicht die eigentliche Frage. Die Frage ist:
> Lernt ein Detektor daraus, echte Tiere zu erkennen? Dafür habe ich zehn Generator-Einstellungen
> verglichen: jeweils 1.200 Bilder für dieselben zwölf Klassen, rechts ein Beispiel, ein
> Löwe. Derselbe Detektor, dasselbe Training, und bewertet wird auf 9.742 echten Fotos.
>
> Blau die API-Modelle, grau die lokalen. Alle API-Modelle liegen vor allen lokalen
> Modellen. Gemini, mit dem ich den Datensatz erzeugt habe, landet auf Platz drei. Der
> optische Eindruck war also kein schlechter, aber auch kein verlässlicher Ratgeber.
>
> Rechts daneben die Kosten pro Bild. Platz eins ist gpt-image-2 in der günstigsten Stufe,
> 1,7 Cent pro Bild. Die doppelt so teure mittlere Stufe ist nicht besser. Bei den lokalen
> Modellen steht dort die Rechenzeit pro Bild auf meiner GPU, einer A40: Das aufwendigste,
> HiDream, braucht 144 Sekunden pro Bild und landet auf dem vorletzten Platz. Der Preis
> sagt also nichts über die Qualität der Trainingsdaten.

**[Klick]** *(nur gpt-image-2 low und Gemini Flash-Lite bleiben)*

### Folie 16 — Training set comparison: kürzerer Prompt · 0:35

> Was dagegen zählt, zeigt dieser Versuch: Zwei der Modelle, jeweils mit vollem Prompt…

**[Klick]** *(Kurzprompt, −45 % und −48 %)*

> …und einmal mit der Kurzfassung von höchstens 75 Tokens. Der Detektor verliert fast die
> Hälfte seiner Genauigkeit, bei beiden Modellen: minus 45 und minus 48 Prozent. Wie viel
> der Generator über ein Tier erfährt, ist wichtiger als die Wahl des Generators.

**[Klick]**

### Folie 17 — Rare animals are a blind spot · 0:50

> Eine wichtige Einschränkung. Das ist die Genauigkeit pro Klasse und Generator, je
> heller, desto besser.

**[Klick]** *(seltene Klassen markiert)*

> Die oberen sechs Zeilen sind seltene Arten, also genau die, für die man generierte
> Bilder am dringendsten braucht. Dort liegt jeder Generator unter 0,07.

**[Klick]** *(Saiga von allen Generatoren)*

> Warum, sieht man hier: die Saiga-Antilope mit ihrer rüsselartigen Nase. Links oben ein
> echtes Foto, daneben Gemini, dann die lokalen Generatoren. Die zeichnen stattdessen eher
> eine gewöhnliche Gazelle, ohne diese Nase. Ein Generator kann nur zeichnen, was er oft
> genug gesehen hat.

**[Klick]**

---

## Akt 4 — Die Klassen ergänzen (`Bands2`)

### Folie 18 — 23,850 synthetic images · 0:20

> Für den eigentlichen Datensatz habe ich mit Gemini 23.850 Bilder erzeugt, zum
> Listenpreis rund 785 Dollar. Hier ein Ausschnitt: Grévyzebras.

**[Klick]**

### Folie 19 — Final Dataset · 1:15

> So sieht der fertige Datensatz aus. Jeder Balken ist eine Klasse, zunächst nur mit ihren
> echten Fotos, sortiert nach der Anzahl.

**[Klick]** *(Bänder A–D)*

> Nach dieser Anzahl teile ich die Klassen in vier Bänder: A sind die 50 Klassen mit den
> wenigsten Fotos, D die gut dokumentierten Arten.

**[Klick]** *(synthetische Trainingsbilder)*

> Jetzt kommen die synthetischen Trainingsbilder dazu, in grün. Band A bekommt nur
> synthetische Bilder, 200 pro Klasse, und alle echten Fotos werden zu Testbildern, hier
> in blau. Band B bekommt halb echte, halb synthetische Bilder. Band C nur echte. A, B und
> C haben also gleich viele Trainingsbilder, je 200, nur die Mischung unterscheidet sich.
> So kann man echte und künstliche Bilder direkt vergleichen. Band D bekommt bis zu 1.500
> echte Bilder.

**[Klick]** *(synthetische Testbilder)*

> Und oben drauf kommen 50 synthetische Testbilder pro Klasse, in gelb.

**[Klick]**

### Folie 20 — Recognition Results · 1:30

> Und das ist das Ergebnis des kleinen Detektors auf echten Fotos, direkt über denselben
> Bändern. Band A, nur synthetisch trainiert: 0,30. Band B, halb synthetisch: 0,59.
> Band C, nur echte Fotos: ebenfalls 0,59. Und Band D mit viel mehr echten Bildern: 0,74.
> Die Hälfte eines kleinen echten Datensatzes lässt sich also durch synthetische Bilder
> ersetzen, ohne messbaren Verlust. Nur synthetisch erreicht man aber nur etwa die Hälfte.
> Synthetische Daten strecken einen echten Datensatz, sie ersetzen ihn nicht.
> Einschränkend: In jedem Band stecken andere Arten.

**[Klick]** *(ein Punkt pro Klasse)*

> Hinter den Durchschnitten steckt viel Streuung. Jeder Punkt ist eine Klasse, an ihrer
> Stelle im Diagramm, der weiße Strich ist der Median pro Band.

**[Klick]** *(Hyäne und Mangaben)*

> In Band A, rein synthetisch trainiert, erreicht die Streifenhyäne 0,48, die Mangaben fast
> null. Meine Vermutung, ausdrücklich als Hypothese: Arten mit einheitlichem Aussehen
> lassen sich gut generieren, sehr vielfältige Klassen wie eine ganze Affengattung nicht.

**[Klick]**

### Folie 21 — Test Data · 0:45

> Noch ein Wort zur Messung selbst. Oben synthetische Testbilder: ein Tier, mittig, freie
> Sicht. Unten echte Fotos: mehrere Tiere, teilweise verdeckt oder weit weg.

**[Klick]** *(Balken)*

> Auf synthetischen Testbildern schneiden alle Modelle viel besser ab, in Band A mit 0,93
> gegenüber 0,20. Selbst das Ensemble, das nie ein generiertes Bild gesehen hat, ist dort
> um etwa 0,2 besser. Die synthetischen Testbilder sind schlicht leichter.
> Deshalb berichte ich alle Werte auf echten Fotos. Nur die sagen etwas über den Einsatz
> im Feld aus.

**[Klick]**

---

## Akt 5 — Modelle und Hardware (`ModelsV2`)

### Folie 22 — Three candidates · 0:20

> Damit zum letzten Schritt: dem Modell. Drei Kandidaten, die Kreisfläche entspricht der
> Parameterzahl: das Ensemble aus MegaDetector und SpeciesNet mit 193 Millionen, YOLOv5s
> mit 7,6 Millionen, die Architektur, die heute schon im Fernglas läuft, und YOLO26n mit
> 2,7 Millionen.

**[Klick]**

### Folie 23 — Accuracy on real photos · 1:00

> Die Genauigkeit auf echten Fotos: Vorne liegt das auf meine Daten feinjustierte Ensemble
> mit 0,60. Der Stern heißt, das ist eine Obergrenze, weil MegaDetector auch die
> Referenzboxen erzeugt hat. YOLO26n kommt auf 0,53. Das ist besser als SpeciesNet ohne
> Anpassung und besser als das größere YOLOv5s.

**[Klick]** *(Größe und Zeit pro Bild)*

> Und jetzt die Zahl, die ich mir am Anfang gemerkt hatte. Gemessen auf einem Raspberry Pi
> 400 als Stellvertreter für den Chip im Fernglas, nur auf der CPU und ohne Quantisierung.
> Das Ensemble braucht etwa 35 Sekunden pro Bild. Bis dahin ist jedes Tier längst weg.
> YOLO26n braucht 0,42 Sekunden.

**[Klick]** *(„83× faster · 71× smaller · −0.07 mAP“)*

> 83-mal schneller, 71-mal kleiner, und dabei nur 0,07 weniger Genauigkeit. Echtzeit ist
> das noch nicht. Dafür fehlen Quantisierung und der Beschleuniger des Chips. Aber wir sind
> von einer halben Minute pro Bild bei unter einer halben Sekunde angekommen.

**[Klick]**

### Folie 24 — Finding is easy. Naming is hard. · 0:40

> Wo verliert das kleine Modell seine Genauigkeit? Ob es das Tier findet, egal welche Art:
> 0,78. Mit der richtigen Art: 0,53.

**[Klick]** *(Elefanten)*

> Das Problem ist also das Benennen, nicht das Finden. Das deutlichste Beispiel: Asiatische
> Elefanten werden 1.545-mal als Afrikanische erkannt, umgekehrt nur neunmal. An der Zahl
> der Testbilder liegt es nicht, von den Asiatischen gibt es sogar mehr.

**[Klick]**

### Folie 25 — Can a big model teach a small one? · 0:50

> Eine naheliegende Idee, um das kleine Modell besser zu machen: Knowledge Distillation.
> Das große Ensemble ist der Lehrer, das kleine Modell der Schüler. In grau der Schüler,
> der vom Lehrer gelernt hat.

**[Klick]**

> In grün derselbe Schüler, einfach direkt auf den Daten trainiert.

**[Klick]**

> Direktes Training gewinnt in jedem Band. Der Grund liegt in den Daten: Die Labels kamen
> von SpeciesNet, also vom Lehrer selbst. Er kann dem Schüler kaum etwas beibringen, was
> nicht schon in den Labels steht. Getestet habe ich allerdings nur eine Konfiguration.

**[Klick]**

---

## Akt 6 — Zusammenfassung (`Summary`)

### Folie 26 — Summary · 1:15

*(Pro Klick erscheint eine Kachel. Die erste ist sofort da.)*

> Zum Schluss der Weg noch einmal im Überblick. Wir haben mit MammalID angefangen, dem
> Ensemble aus MegaDetector und SpeciesNet…

**[Klick]**

> …und den drei Herausforderungen: Daten, Hardware, Geschwindigkeit.

**[Klick]**

> Dann die Daten: zuerst die 225 Klassen…

**[Klick]**

> …für viele davon gibt es zu wenige Bilder…

**[Klick]**

> …und nicht jedes Bild zeigt ein Tier, deshalb die Filterkette.

**[Klick]**

> Dann die generierten Bilder: Gemini sieht besser aus als die lokalen Modelle…

**[Klick]**

> …entscheidend ist, wie viel Prompt ein Generator lesen kann…

**[Klick]**

> …die günstigste Einstellung gewinnt…

**[Klick]**

> …und bei seltenen Tieren stoßen alle Generatoren an ihre Grenzen.

**[Klick]**

> Dann der fertige Datensatz mit seinen vier Bändern…

**[Klick]**

> …mit dem Ergebnis: Die Hälfte der echten Bilder lässt sich ersetzen, aber nicht alle.

**[Klick]**

> Und am Ende ein Modell, das klein und schnell genug für das Gerät ist und trotzdem nah an
> das große Ensemble herankommt. Der nächste Schritt wäre die Optimierung für den Chip,
> damit daraus echte Echtzeit wird.

**[Klick]**

### Folie 27 — Thank you · 0:10

> Vielen Dank für die Aufmerksamkeit. Ich freue mich auf Ihre Fragen.

---

## Vorbereitung auf Fragen

| Frage | Antwort |
|-------|---------|
| Ist SpeciesNet wirklich nicht auf Nashörnern trainiert? | Gemessen habe ich: Von 1.510 Nashornfotos, die SpeciesNet erreicht haben, hat es nur 40 (2,6 %) der richtigen Klasse zugeordnet. Für solche Abdeckungslücken gab es eine Rückfallregel in der Filterkette, die rund 14.500 Bilder aus 17 Klassen gerettet hat. |
| Warum Raspberry Pi statt QCS605? | Der Hexagon-DSP des QCS605 hängt an einer Toolchain von 2018 mit schlecht dokumentierter Operator-Unterstützung. Der Pi 400 ist laut CPU-Vergleich etwa 10–15 % langsamer als die Kryo-CPU des QCS605: 423 ms auf dem Pi entsprechen etwa 360–380 ms auf der CPU des Ziels. Den DSP gibt es auf dem Pi nicht. |
| Wie kommt man auf Echtzeit? | Quantisierung, idealerweise quantisierungsbewusstes Training, dann der Integer-Pfad des Chips. Laut Hersteller schafft YOLO26n auf einem Raspberry Pi 5 etwa 68 ms. Gemessen habe ich das nicht. |
| Warum nur ein Lauf pro Modell? | Jeder Lauf dauerte auf einer A40 zwischen 21 Stunden und 8,8 Tagen. Konfidenzintervalle fehlen deshalb, Unterschiede unter etwa 0,03 sind nicht belastbar. Im Generatorvergleich hat sich ein Abstand dieser Größe mit mehr Seeds sogar umgedreht (0,064 → 0,090 → 0,105). |
| Warum ist das Ensemble eine Obergrenze? | MegaDetector hat die Ground-Truth-Boxen erzeugt. Beim Ensemble ist mAP gleich mAP50, was kein anderer Detektor erreicht. Die Lokalisierung wird also teilweise gegen die eigene Ausgabe bewertet. |
| Warum ursprünglich ein gemischtes Testset? | Die 50 synthetischen Testbilder pro Klasse sollten seltene Klassen mit wenigen echten Testbildern stabilisieren. Die Analyse hat gezeigt, dass sie alle Werte nach oben verzerren, deshalb nur reale Werte. |
| Wurden die synthetischen Bilder automatisch annotiert? | Ja, die Boxen kamen von MegaDetector. Im Generatorvergleich betrifft das alle Konfigurationen gleich, die Rangfolge bleibt gültig, die absoluten Werte sind vorläufig. |
| Warum keine NanoDet, PicoDet usw.? | Aus Zeitgründen. Die Detektorfamilie war durch das Produkt vorgegeben. Ein Architekturvergleich mit Hyperparametersuche ist der naheliegende nächste Schritt. |
| Lizenzen der Generatoren? | Nur zwei lokale Modelle haben wirklich freie Lizenzen, die anderen haben Community- oder nicht-kommerzielle Bedingungen. Die FLUX-API verbietet Training auf ihren Ausgaben. |
| Kosten der Bildgenerierung? | 23.850 Bilder, rund 785 $ zum Listenpreis (die Gemini-Abrechnung bis Mitte Juni zeigte 647,29 €). Für die empfohlenen 1.500 Bilder pro Klasse wären es etwa vierzehnmal so viele. |
