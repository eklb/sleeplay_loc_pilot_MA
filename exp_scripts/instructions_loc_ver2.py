# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:47:34 2022

@author: kolbe
"""
# =============================================================================
# Define all text stimuli
# =============================================================================

fix = "+"

start_text = """
Herzlich Willkommen
\nVielen Dank für die Teilnahme an unserer Studie!
\nBitte nehmen Sie sich etwas Zeit, um sich die folgenden Instruktionen in Ruhe durchzulesen.
\n\n >>> Leertaste für Weiter >>>

"""
pract_text1 = """
Wir beginnen zunächst mit einer kurzen Lernphase.
\nWährend dieser Lernphase werden Ihnen Paare von Bildern und Wörtern präsentiert, die Sie sich für einen anschließenden Gedächtnistest merken sollen.
\nAbhängig von der Sitzung werden die Bilder entweder Objekte oder Szenen zeigen, welche dann zusammen mit einem Wort präsentiert werden.
\n >>> Weiter >>>
"""
pract_text2 = """
Im Gedächtnistest wird Ihnen nur das Wort gezeigt und Sie sollen sich an das zugehörige Bild erinnern. Wenn Sie sich an das zugehörige Bild erinnern, drücken Sie bitte so schnell wie möglich die Leertaste.
\nBitte drücken Sie keine Taste, wenn Sie sich nicht erinnern.
\nSie haben 5 Sekunden Zeit, bis das nächste Wort präsentiert wird.
\n >>> Weiter >>>
"""
pract_text3 = """
Zum Lernen der Bild-Wort-Paare stellen Sie sich das Wort und Bild zusammen möglichst bildhaft vor, auch wenn die Assoziationen absurd erscheinen können.
\nWenn Sie zum Beispiel das Bild eines Vogels und das Verb „treffen“ präsentiert bekommen, könnten Sie sich vorstellen, wie der Vogel von einem Schuss getroffen wird und kopfüber aus dem Bild fällt.
\n >>> Weiter >>>
"""
pract_text4 = """
Wenn Sie zum Beispiel das Bild eines Kanals zusammen mit dem Wort „starr“ sehen, könnten Sie sich vorstellen, wie die Wasseroberfläche erstarrt und eine sichtbar feste Struktur erhält.
\nWerden Sie also möglichst kreativ, um sich die Bild-Wort-Assoziationen gut einprägen zu können.
\n >>> Weiter >>>
"""
pract_text5 = """
Wir starten jetzt mit einer kurzen Übung, damit Sie sich mit der Aufgabe vertraut machen können.
\n\n\n >>> Leertaste für Weiter >>>
"""
pract_break = """
Pause
\nNach jeder Lernphase wird es eine kurze Pause geben.
\nRuhen Sie sich einen Moment aus, bevor wir mit dem Gedächtnistest fortfahren.
\n\nDrücken Sie die Leertaste, wenn Sie bereit sind.
"""
pract_text6 = """
Das war der Übungsblock.
\nHaben Sie noch Fragen?
\n\n >>> Weiter >>>
"""
pract_end = """
Die richtige Aufgabe wird aus mehreren aufeinander folgenden Blöcken bestehen, die ebenfalls eine Lernphase und einen anschließenden Gedächtnistest umfassen.
\nNach jedem zweiten Block wird es die Möglichkeit zu einer Pause (max. 5 min) geben, bevor mit dem nächsten Block gestartet wird.
\nWenn Sie bereit sind, dann geht es jetzt in den Scanner.
\n\nViel Erfolg!
"""


short_break = """Pause
\n\nRuhen Sie sich einen kurzen Moment aus, bevor es mit dem Gedächtnistest weitergehen wird.
"""
fix_break1 = "Super, das war der erste Durchgang! \n\nNach einer kurzen Pause geht es mit dem nächsten Durchgang weiter."
fix_break2 = "Super, ein weiterer Durchgang ist geschafft! \n\nNach einer kurzen Pause geht es mit dem nächsten Durchgang weiter."

# =============================================================================
# block_break = """Super, ein weiterer Block ist geschafft!
# \nEs wird nun eine kurze Pause geben, bevor wir mit dem nächsten Block fortfahren. Bitte nutzen Sie die Zeit, um sich einen Moment auszuruhen. Jede Pause sollte allerdings nicht länger als 5 Minuten dauern.
# \n\n>>> Drücken Sie die Taste, sobald Sie bereit sind fortzufahren.
# """
# =============================================================================

half_time = """
Sehr gut!
\nDie Hälfte ist nun geschafft.
\nWir werden im Anschluss einen anatomischen Scan durchführen, bevor wir mit der zweiten Hälfte der Aufgabe fortfahren.
\nBitte bleiben Sie während des Scans einfach ruhig liegen und bewegen sich möglichst nicht.
\n\n>>> Drücken Sie die Taste, sobald Sie bereit sind fortzufahren.

"""
after_t1 = """
Der anatomische Scan ist abgeschlossen.
\nWenn Sie bereit sind, können wir mit dem nächsten Block starten.
\n\n>>> Drücken Sie die Taste, um mit der Aufgabe fortzufahren.
"""
end_text = """
Super, nun ist der letzte Block geschafft! \nBitte bleiben Sie noch einen Moment liegen, bis wir Sie aus dem Scanner holen.
\n\nVielen Dank für Ihre Teilnahme an unserer Studie.
"""
exit_text = """Das Experiment wurde abgebrochen...
            \nDrücken Sie Escape, um das Fenster zu schließen."""

input_text = "Bitte Wort eintippen und mit Enter bestätigen:"

intro_task = "Odd/Even Task"
task_text1 = "<<<<  ungerade                gerade  >>>>"
task_text2 = "Richtig!"
task_text3 = "Falsch!"

# Color codes
black = (-1,-1,-1)
green = (-0.529411764705882, 0.403921568627451, -0.113725490196078)
red = (1,-1,-1)
light_grey = (0.654901960784314, 0.654901960784314, 0.654901960784314)
    