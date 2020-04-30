# Speaker Recognition

Carmine Romaniello
Università degli studi di Milano, via Celoria 18, Milano, Italy. 
E-mail: carmine.romaniello@studenti.unimi.it

In questo lavoro viene illustrata l’applicazione della Canonical Correlation Analysis (CCA) implementata dalla libreria Pyrcca. Con essa viene eseguita la Cross Modal Analysis per affrontare il problema di Speaker Detection (riconoscimento di chi sta parlando) in contenuti multimediali audio-video. I video utilizzati nella fase di training sono stati reperiti su Youtube, mentre per la fase di test sono stati selezionati alcuni già etichettati reperiti da un dataset già esistente. Dal video in input vengono estratte le features audio e le features video da ogni volto che verranno utilizzate per determinare quello parlante. Per le feature audio vengono estratti 12 MFCC (Mel Frequency Coepstral Coefficent). Per le features video vengono affrontati due approcci: il primo in cui si estraggono le Action Units, il secondo in cui invece si estraggono le distanze tra i Landmark relativi alla bocca. Dai risultati ottenuti con entrambi gli approcci è emerso che il secondo è decisamente migliore.
