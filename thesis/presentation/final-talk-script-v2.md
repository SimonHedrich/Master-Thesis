# Final Talk v2 — Sprechskript

Sprechtext auf Deutsch zur überarbeiteten Fassung der Folien (`slides/talk_v2.py`, Szenen
`UseCase`, `DataV2`, `Generative`, `Bands2`, `ModelsV2`, `Summary`). Die Folien selbst
sind englisch. Jeder **[Klick]** entspricht einem `next_slide()`.

Die Geschichte ist diesmal nicht chronologisch erzählt, sondern als Vorgehen, das Schritt
für Schritt aufeinander aufbaut: Anwendungsfall → Daten → generierte Bilder → Ergänzung
der Klassen → Modelle auf der Hardware → Zusammenfassung.

Alle Genauigkeitswerte sind mAP@[.5:.95] auf **echten** Fotos, außer es steht etwas
anderes dabei. Quellen: `slides/data.py` und Kapitel 4 des Manuskripts.

## Zeitplan

| Akt | Szene | Folien | Zeit | kumuliert |
|-----|-------|--------|------|-----------|
| 1 Anwendungsfall | `UseCase` | 1–7 | 3:30 | 3:30 |
| 2 Daten | `DataV2` | 8–12 | 3:00 | 6:30 |
| 3 Generierte Bilder | `Generative` | 13–21 | 5:30 | 12:00 |
| 4 Klassen ergänzen | `Bands2` | 22–27 | 3:30 | 15:30 |
| 5 Modelle & Hardware | `ModelsV2` | 28–31 | 3:00 | 18:30 |
| 6 Zusammenfassung | `Summary` | 32–33 | 1:30 | 20:00 |

**Wenn die Zeit knapp wird:** Folie 13 (Zeitleiste), Folie 25 (Punktwolke pro Klasse) und
Folie 30 (Elefanten) mit je einem Satz durchklicken. Folie 27 (Testset) und Folie 29
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
> findet, und SpeciesNet von Google, das sie bestimmt. Beide wurden für Kamerafallen
> entwickelt, also für solche Bilder hier.

**[Klick]**

### Folie 5 — Two models, one answer · 0:40

> So arbeiten die beiden zusammen. Hier ein Bild aus einer Kamerafalle.

**[Klick]**

> MegaDetector findet das Tier und zeichnet eine Bounding Box, mit 94 Prozent Konfidenz.
> Der Ausschnitt geht an SpeciesNet, und das sagt: Ozelot, mit 98,8 Prozent.

**[Klick]**

> Zusammen sind das rund 193 Millionen Parameter. Das merken wir uns für später.

**[Klick]**

### Folie 6 — Built for camera traps · 0:45

> Das Problem: SpeciesNet ist für Kamerafallen gebaut, also für nächtliche Infrarotbilder
> aus festen Kameras. Durch ein Fernglas sieht man aber etwas anderes: Tageslicht, große
> Entfernung, und andere Tiere.

**[Klick]**

> Ein Beispiel sind Nashörner. Die sind in den Trainingsdaten von SpeciesNet praktisch
> nicht vertreten. In meinen Daten hat SpeciesNet von 1.510 Nashornfotos gerade einmal
> 40 als Nashorn erkannt.

**[Klick]**

> Meine Hoffnung war deshalb: Ein Modell, das gezielt auf diesen Anwendungsfall trainiert
> wird, also auf Säugetiere, die man mit einem Fernglas beobachtet, kann für die Nutzer
> bessere Ergebnisse liefern.

**[Klick]**

### Folie 7 — Three challenges · 0:30

> Dafür gibt es drei Herausforderungen. Erstens die Daten: 225 Säugetierklassen, von
> denen viele kaum fotografiert werden. Zweitens die Hardware: ein mobiler Chip, offline.
> Und drittens die Geschwindigkeit: Die Antwort muss kommen, solange das Tier noch im Blick
> ist.
> Und damit fängt man an, wo jedes solche Projekt anfängt: bei den Daten.

**[Klick]**

---

## Akt 2 — Daten (`DataV2`)

### Folie 8 — Which animals? · 0:30

> Zuerst muss man festlegen, was das Modell überhaupt erkennen soll. SpeciesNet kennt rund
> zweieinhalbtausend Arten, sortiert nach der biologischen Hierarchie von Stamm bis Art.

**[Klick]**

> Für das Fernglas braucht man davon nur die Säugetiere, die man draußen tatsächlich durch
> ein Fernglas sieht.

**[Klick]**

### Folie 9 — Igel · 0:25

> Die zweite Frage ist, wie fein man unterscheiden will.

**[Klick]** *(„17 species → 1 class“)*

> Das hier sind siebzehn Igelarten. Aus fünfzig Metern Entfernung kann die niemand
> auseinanderhalten. Daraus wird also eine Klasse. Deshalb sind manche Klassen Arten,
> andere Gattungen oder ganze Familien.

**[Klick]**

### Folie 10 — 225 classes · 0:15

> Am Ende stehen 225 Klassen, die meisten davon Raubtiere und Paarhufer. Der Product Owner
> bei Swarovski hat diese Auswahl abgenommen.

**[Klick]**

### Folie 11 — Images per class · 0:45

> Als Nächstes braucht man Bilder für jede Klasse. Startpunkt waren Fotos aus GBIF, einer
> großen Biodiversitätsdatenbank. Jeder Balken ist eine Klasse, die Achse ist
> logarithmisch: 114 Klassen haben weniger als hundert Bilder, 30 gar keine.

**[Klick]** *(weitere Quellen)*

> Also kamen vier weitere Quellen dazu, und zwar nur Bilder mit Lizenzen, die eine
> kommerzielle Nutzung erlauben. Das hilft vielen Klassen. Aber 29 Klassen bleiben unter
> hundert Bildern, und für viele weitere reicht es nur knapp.

**[Klick]**

### Folie 12 — Is this an animal? · 0:45

> Und nicht jedes Bild, das als Tier markiert ist, zeigt auch eines. Das sind echte
> Wikimedia-Bilder aus den Kategorien Gepard, Schimpanse und Tiger: eine ägyptische
> Wandmalerei, ein Affe an der Schreibmaschine, ein Uni-Maskottchen, eine
> Verbreitungskarte.

**[Klick]** *(Filterkette)*

> Deshalb läuft jedes Bild durch eine Filterkette: Metadaten, Bildqualität, MegaDetector,
> ein Vision-Language-Modell als zweite Meinung und am Ende SpeciesNet. Die unklaren
> Fälle, knapp achtzehntausend Bilder, habe ich von Hand geprüft.

**[Klick]** *(SpeciesNet hervorgehoben)*

> Wichtig für später: Die Artenlabels stammen damit von SpeciesNet.
> Und es bleibt das Problem vom Anfang: Für viele Klassen gibt es einfach zu wenige
> Fotos. Also muss man sich überlegen, wie man sonst an Bilder kommt.

**[Klick]**

---

## Akt 3 — Generierte Bilder (`Generative`)

### Folie 13 — Image generation keeps getting better · 0:25

> Die naheliegende Idee: Bilder generieren. Bildgeneratoren sind in den letzten Jahren
> enorm besser geworden, von Stable Diffusion 2022 über SDXL, SD 3 und FLUX bis zu den
> aktuellen Modellen wie Gemini oder gpt-image-2.

**[Klick]**

### Folie 14 — Lokale Modelle vs. Gemini · 0:45

> Hier ein Binturong, ein Marderbär. Links echte Fotos.

**[Klick]** *(lokale Modelle)*

> Daneben drei Generatoren, die man lokal auf einer eigenen GPU laufen lassen kann:
> RealVisXL, Stable Diffusion 3.5 und FLUX. Man sieht: Das sind eher Bären als
> Binturongs.

**[Klick]** *(Gemini)*

> Und hier Gemini, ein kommerzielles Modell über eine API. Optisch ist der Unterschied
> deutlich. Mit Gemini habe ich deshalb den eigentlichen Datensatz erzeugt.

**[Klick]**

### Folie 15 — Three zebras · 0:20

> Wie genau die Bilder sein müssen, zeigen Zebras: Grévyzebra, Bergzebra, Steppenzebra.
> Der Unterschied liegt in den Streifen. Genau diese Details muss der Generator treffen.

**[Klick]**

### Folie 16 — Der Prompt · 0:30

> Deshalb sind die Prompts lang. Das hier ist ein Prompt für ein einziges Bild: die
> Artbeschreibung aus Wikipedia, Lebensraum, Verhalten und eine Szene mit Pose, Umgebung,
> Licht und Entfernung.

**[Klick]**

> Dazu die Merkmale, an denen man die Art erkennt, beim Grévyzebra etwa die schmalen
> Streifen und der weiße Bauch. Im Schnitt etwa 1.300 Wörter pro Bild.

**[Klick]**

### Folie 17 — How much does the generator read? · 0:40

> Das Problem: Nicht jeder Generator liest so viel. Hier derselbe Löwe mit vier
> verschiedenen Prompts. Ganz links eine Kurzfassung mit höchstens 75 Tokens, das ist
> alles, was ältere Text-Encoder lesen. Dann Fassungen mit 256 und 512 Tokens für die
> neueren lokalen Modelle. Und ganz rechts der volle Prompt mit über viertausend Wörtern,
> den nur die API-Modelle verarbeiten.
> Man muss den Text nicht lesen können. Wichtig ist der Unterschied in der Länge.

**[Klick]**

### Folie 18 — Der Versuchsaufbau · 0:35

> Ob die Bilder optisch gut aussehen, ist aber nicht die eigentliche Frage. Die Frage ist:
> Lernt ein Detektor daraus, echte Tiere zu erkennen? Dafür habe ich acht Quellen
> verglichen, hier am Beispiel Löwe.

**[Klick]** *(Aufbau)*

> Zwölf Konfigurationen, jeweils 1.200 Bilder für dieselben zwölf Klassen, derselbe
> Detektor, dasselbe Training, und bewertet wird auf 9.742 echten Fotos.

**[Klick]**

### Folie 19 — Ranking · 0:45

> Das Ergebnis: Blau die API-Modelle, grau die lokalen. Alle API-Modelle liegen vor allen
> lokalen Modellen.

**[Klick]** *(„production choice: #3“)*

> Gemini, mit dem ich den Datensatz erzeugt habe, landet auf Platz drei. Der optische
> Eindruck war also kein schlechter, aber auch kein verlässlicher Ratgeber.

**[Klick]** *(„#1 = cheapest setting“)*

> Platz eins ist gpt-image-2 in der günstigsten Stufe. Die fast dreimal so teure mittlere
> Stufe ist nicht besser, und bei den lokalen Modellen landet das aufwendigste, HiDream,
> auf dem vorletzten Platz. Der Preis sagt also nichts über die Qualität der
> Trainingsdaten.

**[Klick]**

### Folie 20 — Shorter prompt, worse data · 0:30

> Was dagegen zählt, zeigt dieser Versuch: dasselbe Modell, einmal mit vollem Prompt…

**[Klick]** *(Kurzprompt)*

> …und einmal mit der Kurzfassung. Der Detektor verliert fast die Hälfte seiner
> Genauigkeit, bei beiden Modellen. Wie viel der Generator über ein Tier erfährt, ist
> wichtiger als die Wahl des Generators.

**[Klick]**

### Folie 21 — Rare animals are a blind spot · 0:45

> Eine wichtige Einschränkung. Das ist die Genauigkeit pro Klasse und Generator, je
> heller, desto besser.

**[Klick]** *(seltene Klassen markiert)*

> Die oberen sechs Zeilen sind seltene Arten, also genau die, für die man generierte
> Bilder am dringendsten braucht. Dort liegt jeder Generator unter 0,07.

**[Klick]** *(Saiga)*

> Warum, sieht man hier: die Saiga-Antilope mit ihrer rüsselartigen Nase. Die lokalen
> Generatoren zeichnen stattdessen eine gewöhnliche Gazelle. Ein Generator kann nur
> zeichnen, was er oft genug gesehen hat.

**[Klick]**

---

## Akt 4 — Die Klassen ergänzen (`Bands2`)

### Folie 22 — 23,850 synthetic images · 0:15

> Für den eigentlichen Datensatz habe ich mit Gemini 23.850 Bilder erzeugt, zum
> Listenpreis rund 785 Dollar.

**[Klick]**

### Folie 23 — Same budget, different mix · 0:40

> Verteilt wurden sie nach Datenlage, in vier Bänder. Jedes Quadrat steht für zehn
> Trainingsbilder, grün umrandet heißt synthetisch.
> Band A, die 50 Klassen mit den wenigsten Fotos, bekommt nur synthetische Bilder. Band B
> halb echte, halb synthetische. Band C nur echte. Und Band D, die gut dokumentierten
> Arten, bis zu 1.500 echte Bilder.

**[Klick]** *(„200 images each“)*

> A, B und C bekommen gleich viele Bilder, je 200. Nur die Mischung unterscheidet sich.
> So kann man echte und künstliche Bilder direkt vergleichen. Einschränkend: In jedem Band
> stecken andere Arten.

**[Klick]**

### Folie 24 — Every class, train and test · 0:30

> Hier das Ganze für jede einzelne Klasse: hellgrau echte Trainingsbilder, grün
> synthetische, blau die echten Testbilder, und blassgrün oben 50 synthetische Testbilder
> pro Klasse. Man sieht schön, wie die Bänder von links nach rechts immer mehr echte Bilder
> bekommen.

**[Klick]**

### Folie 25 — Ergebnisse pro Band · 0:40

> Und das ist das Ergebnis des kleinen Detektors auf echten Fotos. Band B, halb
> synthetisch: 0,59. Band C, nur echte Fotos: ebenfalls 0,59.

**[Klick]** *(„=“ und „½“)*

> Die Hälfte eines kleinen echten Datensatzes lässt sich also durch synthetische Bilder
> ersetzen, ohne messbaren Verlust. Nur synthetisch, in Band A, erreicht man aber nur etwa
> die Hälfte. Synthetische Daten strecken einen echten Datensatz, sie ersetzen ihn nicht.

**[Klick]**

### Folie 26 — Not every animal is the same · 0:30

> Hinter den Durchschnitten steckt viel Streuung. Jeder Punkt ist eine Klasse, der weiße
> Strich der Median.

**[Klick]** *(Hyäne und Mangaben)*

> In Band A, rein synthetisch trainiert, erreicht die Streifenhyäne 0,48, die Mangaben fast
> null. Meine Vermutung, ausdrücklich als Hypothese: Arten mit einheitlichem Aussehen
> lassen sich gut generieren, sehr vielfältige Klassen wie eine ganze Affengattung nicht.

**[Klick]**

### Folie 27 — Was the test fair? · 0:45

> Noch ein Wort zur Messung selbst. Links ein synthetisches Testbild: ein Tier, mittig,
> freie Sicht. Rechts ein echtes: mehrere Tiere, teilweise verdeckt.

**[Klick]** *(Balken)*

> Auf synthetischen Testbildern schneiden alle Modelle viel besser ab, in Band A mit 0,93
> gegenüber 0,20. Selbst das Ensemble, das nie ein generiertes Bild gesehen hat, ist dort
> um etwa 0,2 besser. Die synthetischen Testbilder sind schlicht leichter.

**[Klick]**

> Deshalb berichte ich alle Werte auf echten Fotos. Nur die sagen etwas über den Einsatz
> im Feld aus.

**[Klick]**

---

## Akt 5 — Modelle und Hardware (`ModelsV2`)

### Folie 28 — Three candidates · 0:25

> Damit zum letzten Schritt: dem Modell. Drei Kandidaten, die Kreisfläche entspricht der
> Parameterzahl: das Ensemble aus MegaDetector und SpeciesNet mit 193 Millionen, YOLOv5s
> mit 7,6 Millionen und YOLO26n mit 2,7 Millionen.

**[Klick]**

> YOLOv5s ist die Architektur, die heute schon im Fernglas läuft.

**[Klick]**

### Folie 29 — Accuracy on real photos · 1:00

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

### Folie 30 — Finding is easy. Naming is hard. · 0:40

> Wo verliert das kleine Modell seine Genauigkeit? Ob es das Tier findet, egal welche Art:
> 0,78. Mit der richtigen Art: 0,53.

**[Klick]** *(Elefanten)*

> Das Problem ist also das Benennen, nicht das Finden. Das deutlichste Beispiel: Asiatische
> Elefanten werden 1.545-mal als Afrikanische erkannt, umgekehrt nur neunmal. An der Zahl
> der Testbilder liegt es nicht, von den Asiatischen gibt es sogar mehr.

**[Klick]**

### Folie 31 — Can a big model teach a small one? · 0:50

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

### Folie 32 — Summary · 1:15

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

> Dann die Bänder, mit gleichem Budget und unterschiedlicher Mischung…

**[Klick]**

> …mit dem Ergebnis: Die Hälfte der echten Bilder lässt sich ersetzen, aber nicht alle.

**[Klick]**

> Und am Ende ein Modell, das klein und schnell genug für das Gerät ist und trotzdem nah an
> das große Ensemble herankommt. Der nächste Schritt wäre die Optimierung für den Chip,
> damit daraus echte Echtzeit wird.

**[Klick]**

### Folie 33 — Thank you · 0:10

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
