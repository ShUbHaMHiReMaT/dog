"""Simple-English PDF report of the 22 Oct 2026 wrist trial -> reports/PRANIIVA_Wrist_Trial_Report.pdf"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "PRANIIVA_Wrist_Trial_Report.pdf"
IMG = ROOT / "reports" / "sessions" / "MD_WRIST" / "session_overview.png"

TEAL, INK, MUTED, LINE = colors.HexColor("#0f6f68"), colors.HexColor("#16232b"), colors.HexColor("#55656f"), colors.HexColor("#d9e0e4")
GOOD, WARN = colors.HexColor("#e6f3ea"), colors.HexColor("#fbf0dc")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=INK, alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontName="Helvetica", fontSize=10, leading=14, textColor=MUTED)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=TEAL, spaceBefore=12, spaceAfter=6)
P = ParagraphStyle("P", parent=ss["Normal"], fontName="Helvetica", fontSize=10.5, leading=15, textColor=INK, spaceAfter=6)
SMALL = ParagraphStyle("S", parent=P, fontSize=9, leading=12, textColor=MUTED)
CELL = ParagraphStyle("C", parent=P, fontSize=9.5, leading=13, spaceAfter=0)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="Helvetica-Bold")
BOX = ParagraphStyle("BOX", parent=P, fontSize=11, leading=16)


def table(rows, widths, header=True, shade=None):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f4")))
    for r, c in (shade or []):
        st.append(("BACKGROUND", (0, r), (-1, r), c))
    t.setStyle(TableStyle(st))
    return t


def bullets(items):
    return ListFlowable([ListItem(Paragraph(x, P), leftIndent=12) for x in items], bulletType="bullet", start="-",
                        leftIndent=14)


def numbered(items):
    return ListFlowable([ListItem(Paragraph(x, P), leftIndent=12) for x in items], bulletType="1", leftIndent=16)


def boxed(text, bg):
    t = Table([[Paragraph(text, BOX)]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def footer(c, doc):
    c.saveState()
    c.setFont("Helvetica", 8)
    c.setFillColor(MUTED)
    c.drawString(20 * mm, 12 * mm, "PRANIIVA prototype - wrist trial report - 22 October 2026 - research prototype, not a medical device")
    c.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    c.restoreState()


s = []
s += [Paragraph("PRANIIVA Wrist Trial Report", H1),
      Paragraph("First test of the real device on a person - 22 October 2026", SUB),
      Spacer(1, 10),
      boxed("<b>Result in one sentence:</b> the device recorded reliably for 36 minutes and is ready for the first dog "
            "trial, once the 4 wires to the motion sensor are soldered.", GOOD)]

s += [Paragraph("1. What the device is", H2),
      Paragraph("A small electronic board with two sensors:", P),
      bullets(["<b>Motion sensor (BMI270)</b> - measures movement and tilt 50 times every second. It feels gravity, so it "
               "knows which way is down, and it feels every step, shake and jump.",
               "<b>Microphone (INMP441)</b> - measures only how LOUD it is, twice a second. It never records voices or sound.",
               "<b>The brain (ESP32-S3)</b> - reads both sensors and sends the numbers to the laptop through a USB cable."]),
      Paragraph("On the laptop, every 2 seconds the software decides what the wearer is doing (still, small movements, "
                "walking, running, shaking, and for a dog also standing, sitting and lying). It shows this live on a website "
                "with a 3D dog, and it saves everything to files.", P)]

s += [Paragraph("2. What we did", H2),
      Paragraph("Our founder wore the device on his wrist for about 27 minutes while sitting, walking and shaking his hand. "
                "Then we put it on a table and turned it into 8 positions to calibrate it. The whole session lasted 36 minutes "
                "and was recorded live.", P)]

s += [Paragraph("3. Main results", H2),
      table([["What we checked", "Result", "Good?"],
             ["How long it recorded", "36 minutes without stopping", "Yes"],
             ["Data received", "108,378 of 108,514 readings (99.87%)", "Yes"],
             ["USB cable disconnections", "0", "Yes"],
             ["Microphone", "Worked 100% of the time", "Yes"],
             ["Strongest movement", "2.8 g (the sensor can measure up to 8 g, so nothing was cut off)", "Yes"],
             ["All files saved", "Movement, sound, labels, notes, device health, session info", "Yes"],
             ["Data lost", "About 2 seconds, while taking it off the hand (a loose wire)", "Needs a fix"]],
            [52 * mm, 92 * mm, 26 * mm], shade=[(7, WARN)])]

if IMG.exists():
    s += [Spacer(1, 8), Image(str(IMG), width=170 * mm, height=170 * mm * 931 / 1430),
          Paragraph("Top: how much the device moved. Middle: what the system decided every 2 seconds. Bottom: how loud it "
                    "was. He wore it for the first 27.5 minutes; after that it lay on the table for calibration.", SMALL)]

s += [PageBreak(), Paragraph("4. What the system saw on his wrist", H2),
      table([["Activity", "Time"],
             ["Small movements (hand gestures, typing, talking)", "10.4 minutes"],
             ["Still (sitting with the arm resting)", "9.1 minutes"],
             ["Walking", "6.5 minutes"],
             ["Large slow movements (big arm swings)", "50 seconds"],
             ["Shaking (fast hand shakes)", "18 seconds"],
             ["Running", "16 seconds"]], [120 * mm, 50 * mm]),
      Spacer(1, 4),
      Paragraph("Note: the system is built for a dog's body, so on a person's wrist these labels only show that the system "
                "works. We did not write down what he was doing at each moment, so we cannot yet say how accurate it is.", SMALL)]

s += [Paragraph("5. Calibration (making the sensor more exact)", H2),
      Paragraph("Every motion sensor has tiny built-in errors. We placed the device in 8 positions on the table, and the "
                "software worked out these errors and now corrects them automatically. This is the same method used in "
                "research (called GGIR, van Hees 2014).", P),
      table([["Measure", "Result"],
             ["Error before calibration", "6.3 thousandths of gravity"],
             ["Error after calibration", "0.8 thousandths of gravity - 8 times more exact"],
             ["Sensor health", "Very small errors - the sensor is healthy"],
             ["Speed of readings", "49.99 per second (should be 50) - excellent"]], [60 * mm, 110 * mm])]

s += [Paragraph("6. 11,440 automatic tests", H2),
      Paragraph("We also tested the software 11,440 times with made-up data, including the worst situations reported in "
                "research papers about motion sensors worn on the body. <b>All 11,440 tests passed.</b> Every test is saved, so "
                "any problem can be checked again.", P),
      table([["Type of test", "What it checks", "Passed"],
             ["Movements", "Still, small, walking, running, shaking - with the device turned in any direction", "4,000 of 4,000"],
             ["Postures", "Standing, sitting, lying on belly, on side, on back", "2,400 of 2,400"],
             ["Bad data", "Broken messages, missing data, the cable being pulled out", "1,500 of 1,500"],
             ["Saved files", "Every reading saved once and in the right format", "1,500 of 1,500"],
             ["Worst cases", "Sensor glitches, car rides, very short shakes, very hard play, heat changes, collar "
                             "turning, loose strap, stuck sensor", "1,500 of 1,500"],
             ["Fit check", "Device on a table, worn, loose, tight, moved, sensor fault", "540 of 540"]],
            [30 * mm, 105 * mm, 35 * mm])]

s += [PageBreak(), Paragraph("7. Mistakes we found and fixed", H2),
      table([["Problem we found", "How we fixed it"],
             ["Big slow arm swings were called 'running'", "Running now needs a real running rhythm. Big slow movements get "
                                                           "their own name."],
             ["One bad reading could make 'still' look like movement", "Single bad readings are now removed first."],
             ["Shaking in a car could look like the dog moving", "Car and engine shaking is now ignored."],
             ["The clock could repeat at the start of a session", "The time now always moves forward."],
             ["A new warning message repeated itself", "Fixed, and a test now checks for it."]],
            [75 * mm, 95 * mm])]

s += [Paragraph("8. New: placement check", H2),
      Paragraph("The website now tells you if the device is placed correctly:", P),
      bullets(["<b>On the body?</b> - a living body always makes tiny movements; a table does not.",
               "<b>Strap tight?</b> - a loose device rattles while walking.",
               "<b>Moved since calibration?</b> - checks the device has not slipped.",
               "<b>Sensor reading correct?</b> - gravity must read exactly 1."]),
      Paragraph("On this trial it was right both times: 'worn and tight' on his wrist, and 'not on the body' about 2 minutes "
                "after the device was put on the table.", P)]

s += [Paragraph("9. The one real problem", H2),
      boxed("Twice, about 2 seconds of data were lost when the device was <b>handled</b>. The jumper wires to the motion "
            "sensor wiggle and lose contact for a moment. The device fixes itself, but on a running, shaking dog this would "
            "happen often. <b>Fix: solder the 4 wires (3V3, GND, SDA, SCL).</b>", WARN)]

s += [Paragraph("10. What is not proven yet", H2),
      bullets(["A person's wrist is not a dog's chest. Posture (sitting, lying) and breathing rate can only be tested on a dog.",
               "We do not yet know how accurate it is on a real dog. For that we need notes of what the dog is really doing.",
               "The battery and the SD memory card do not work yet. For now the device runs from USB."])]

s += [KeepTogether([Paragraph("11. Next steps before the dog trial", H2),
      numbered(["Solder the 4 wires to the motion sensor.",
                "Fix the device firmly on a chest harness. Power it from a USB power bank.",
                "Open the website and start a session. Press 'calibrate standing' while the dog stands, then 'calibrate "
                "sitting' while it sits.",
                "While the dog walks, sits, lies down and shakes, type what it is doing as notes (markers) on the website.",
                "Send us the session. We will measure the real accuracy and improve the system."])])]

s += [Spacer(1, 10), Paragraph("Where the data is kept: data/raw/MD_WRIST (recording), reports/sessions/MD_WRIST (analysis "
                               "and calibration), reports/test_campaign (all 11,440 test results).", SMALL)]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                        bottomMargin=20 * mm, title="PRANIIVA Wrist Trial Report", author="PRANIIVA")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print(OUT)
