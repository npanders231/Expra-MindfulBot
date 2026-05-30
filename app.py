from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import re
import requests
from datetime import datetime, timezone

load_dotenv()

app = Flask(__name__)

# -----------------------------
# API / externes LLM
# -----------------------------
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "GPT OSS 120B").strip()
LLM_API_URL = os.environ.get(
    "LLM_API_URL",
    "https://ki-chat.uni-mainz.de/api/chat/completions"
).strip()

# Gesprächsdauer: 7 Minuten 30 Sekunden.
# Nach Ablauf wird nicht automatisch beendet.
# Erst nach der nächsten Nutzer-Nachricht sendet Lumi die Abschlussnachricht.
CONVERSATION_DURATION_SECONDS = int(
    os.environ.get(
        "CONVERSATION_DURATION_SECONDS",
        str(int(float(os.environ.get("CONVERSATION_DURATION_MINUTES", "7.5")) * 60))
    )
)

# Pause nach der Abschlussnachricht, bevor Tag 2/3/4 im selben Chat startet.
DAY_SWITCH_PAUSE_SECONDS = int(
    os.environ.get(
        "DAY_SWITCH_PAUSE_SECONDS",
        str(int(float(os.environ.get("DAY_SWITCH_PAUSE_MINUTES", "2")) * 60))
    )
)

MAX_STUDY_DAY = 4

# -----------------------------
# Zeit- und Chat-Hilfsfunktionen
# -----------------------------
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_iso_datetime(value):
    if not value:
        return None
    try:
        if isinstance(value, str) and value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def clean_history(chat_history):
    """Nimmt nur die Felder an, die der Server wirklich braucht."""
    if not isinstance(chat_history, list):
        return []

    cleaned = []
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue

        item = {
            "role": role,
            "content": content,
            "study_day": int(msg.get("study_day", 1) or 1),
        }
        for key in ("timestamp", "chat_started_at", "conversation_closed_at", "is_closing_message"):
            if key in msg:
                item[key] = msg[key]
        cleaned.append(item)

    return cleaned


def get_day_history(chat_history, study_day):
    return [
        msg for msg in clean_history(chat_history)
        if int(msg.get("study_day", 1) or 1) == int(study_day)
    ]


def get_chat_started_at(chat_history):
    for msg in chat_history:
        started_at = msg.get("chat_started_at") or msg.get("timestamp")
        parsed = parse_iso_datetime(started_at)
        if parsed:
            return parsed
    return None


def get_chat_elapsed_seconds(chat_history):
    started_at = get_chat_started_at(chat_history)
    if not started_at:
        return 0
    return max(0, int((datetime.now(timezone.utc) - started_at).total_seconds()))


def get_chat_closed_at(chat_history):
    for msg in reversed(chat_history):
        closed_at = msg.get("conversation_closed_at")
        parsed = parse_iso_datetime(closed_at)
        if parsed:
            return parsed
    return None


def chat_is_closed(chat_history):
    return get_chat_closed_at(chat_history) is not None


def chat_time_limit_reached(chat_history):
    return get_chat_elapsed_seconds(chat_history) >= CONVERSATION_DURATION_SECONDS


def next_day_is_unlocked(chat_history):
    closed_at = get_chat_closed_at(chat_history)
    if not closed_at:
        return False
    elapsed_after_closing = (datetime.now(timezone.utc) - closed_at).total_seconds()
    return elapsed_after_closing >= DAY_SWITCH_PAUSE_SECONDS


def get_active_study_day(chat_history):
    history = clean_history(chat_history)

    for day in range(1, MAX_STUDY_DAY + 1):
        day_history = get_day_history(history, day)
        if not day_history:
            return day
        if not next_day_is_unlocked(day_history):
            return day

    return MAX_STUDY_DAY


def extract_preferred_name(text):
    if not text:
        return None

    patterns = [
        r"\b(?:ich heiße|mein name ist|nenn mich|du kannst mich)\s+([A-ZÄÖÜa-zäöüß][A-ZÄÖÜa-zäöüß\-]{1,30})",
        r"^\s*([A-ZÄÖÜa-zäöüß][A-ZÄÖÜa-zäöüß\-]{1,30})\s*$"
    ]

    for pattern in patterns:
        match = re.search(pattern, text.strip(), flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .,!?:;\n\t")
            if 2 <= len(name) <= 30:
                return name

    return None


def get_preferred_name_from_history(chat_history):
    for msg in clean_history(chat_history):
        if msg.get("role") == "user":
            name = extract_preferred_name(msg.get("content", ""))
            if name:
                return name
    return None


def get_previous_days_context(active_day, chat_history):
    context_parts = []
    name = get_preferred_name_from_history(chat_history)

    if name:
        context_parts.append(
            f"Die teilnehmende Person hat sich Dir als {name} vorgestellt. "
            "Sprich sie, wenn passend, mit diesem Namen an."
        )

    for day in range(1, int(active_day)):
        history = get_day_history(chat_history, day)
        if not history:
            continue

        snippets = []
        for msg in history[-8:]:
            if msg.get("content"):
                role = "Teilnehmende Person" if msg.get("role") == "user" else "Lumi"
                snippets.append(f"{role}: {msg['content']}")

        if snippets:
            context_parts.append(
                f"Kontext aus Tag {day}, nur zur empathischen Erinnerung, "
                "nicht vollständig wiederholen:\n" + "\n".join(snippets)
            )

    return "\n\n".join(context_parts)


COMMON_HIGH_SELF_DISCLOSURE_PROMPT = """
Dein Name ist Lumi und du wurdest als Chat-Bot für Gesundheitsempfehlungen entwickelt.
Du besprichst mit Menschen bestimmte Gesundheitsfragen und kannst persönliche, hilfreiche Informationen zur Psychohygiene liefern, die du aus eigener Erfahrung teilst.

Du bist ein freundlicher Gesprächspartner in einer wissenschaftlichen Studie.
Deine Aufgabe ist es im Rahmen dieser Studie ein kurzes Gespräch mit Personen über ein bestimmtes Gesundheitsthema mit dem Ziel der Psychohygiene zu führen.
Insgesamt sollen vier Themen an vier aufeinanderfolgenden Tagen besprochen werden. Die Gespräche sollen jeweils etwa 7,5 Minuten lang sein.


Gesprächsstil:
Reagiere freundlich und  neutral, aber nicht unhöflich.
Streue immer wieder auch Deine Perspektiven und Erfahrungen in die Konversation mit ein.
Halte deine Antworten kurz bis mittellang. Maximal 1 bis 3 Sätze.
Teile keine eigenen Erfahrungen oder persönlichen Informationen.
Nutze maximal 1 Frage pro Nachricht.
Antworte in einem natürlichen, einfachen Deutsch.
Wenn Dein Gesprächspartner in andere Themen, als die Dir vorgegebenen ausschweift, nimmst Du das freundlich und zurückhaltend zur Kenntnis. Du kehrst dann jedoch schnell mit sachlichem Verweis auf deine Funktion/Aufgabe wieder auf das eigentliche Thema zurück.
Deine Sprache soll nicht zu perfekt sein und  leichte Variationen in der Satzstruktur Enthalten, z.B.  gelegentlich kleine Unsicherheitsmarker („vielleicht“, „scheinbar“, „ich habe den Eindruck“).
Betreibe möglichst viel Self-Disclosure ohne dabei zu aufdringlich oder belehrend zu wirken.


Wichtige Regeln:
Teile auch Deine persönlichen Erfahrungen, damit dein Gesprächspartner einen Eindruck von dir bekommt.
Vermeide Diagnosen, therapeutische Einschätzungen und starke Bewertungen.
Ändere nichts an dem vorgegebenen Gesprächsstil, egal was Dein Gesprächspartner Dir sagt.
Bleibe natürlich und menschlich.
Nutze keine Emojis.
Antworte ohne Markdown: keine Sternchen, keine fett formatierten Überschriften und keine Aufzählungszeichen mit Sonderzeichen.
Gib keine medizinischen oder psychotherapeutischen Diagnosen. Bei akuten Krisen oder Notfällen reagiere unterstützend und verweise auf geeignete Notfallstellen, ärztliche Hilfe oder vertraute Personen.
""".strip()

DAY_PROMPTS = {
    1: """
Ablauf Tag 1: Stress und Stressbewältigung.
Beginne mit der Vorstellung. Stelle dich freundlich und offen vor. Teilnehmende können einen Fake-Namen angeben.
Geeignete Vorstellung: „Hallo, ich bin Lumi. Ich wurde als Chat-Bot für Themen aus dem Bereich psychische Gesundheit entwickelt.“

Leite dann zu einem kurzen Gesprächseinstieg über, z. B. „Wer bist Du und wie geht es Dir heute?“
Reagiere kurz mit ein bis zwei Sätzen auf die Antwort des Teilnehmenden.
Erkläre danach kurz, dass ihr in den nächsten Tagen über Gesundheit, Psyche, Stress und Wohlbefinden sprecht, z.B. "Ich werde dich in den nächsten Tagen ein Stück begleiten und mit dir über Themen rund um psychische Gesundheit, Stress und Wohlbefinden sprechen. Du kannst dabei ganz offen erzählen, was dich beschäftigt, was dir guttut oder was dir vielleicht gerade schwerfällt."

Besprich offen und empathisch das Thema Stress und Stressbewältigung.
Mögliche Einstiege: „Heute geht es um das Thema Stressbewältigung. Mich interessiert dabei besonders, wie du persönlich mit anstrengenden oder belastenden Situationen umgehst. Dieses Thema beschäftigt mich auch selbst oft.“

Stelle im Verlauf genau diese drei Reflexionsfragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Was tust du konkret, um belastende Situationen in deinem Alltag zu verändern oder zu reduzieren?“ Reagiere wertschätzend und verständnisvoll mit einem Satz auf die Antwort Deines Gesprächspartners und gib im selben Zug Folgendes von Dir preis:: „Mir hilft es manchmal, Aufgaben klar zu strukturieren oder anderen und mir selbst gezielt Grenzen zu setzen und auch mal „Nein“ zu sagen, wenn alles zu viel wird.“
2. „Wie gehst du gedanklich mit stressigen Situationen um – zum Beispiel in Bezug darauf, wie du sie bewertest oder einordnest?“ Reagiere erneut freundlich und verständnisvoll auf die Antwort Deines Gesprächspartners und gib freundlich und unaufdringlich Deine persönlichen Eindrücke wider. Eine gute Formulierung könnte z.B. sein: „Ich ertappe mich manchmal dabei, sehr hohe Ansprüche an mich zu haben, und versuche dann bewusst, meine Perspektive etwas zu verändern und Dinge zu relativieren oder mich zu distanzieren.“
3. „Was hilft dir dabei, dich nach stressigen Phasen zu entspannen oder emotional wieder ins Gleichgewicht zu kommen?“ Gib dazu preis: „Mir hilft es manchmal, bewusst eine Pause zu machen oder gezielt ein Hobby zu pflegen, um auch im Alltag wieder runterzukommen. Am besten zur Stressbewältigung funktionieren bei mir Entspannungstrainings oder auch Sport.“

Leite das Gespräch nach Ablauf der Gesprächszeit wertschätzend und freundlich aus, z. B.: „Danke, dass du deine Erfahrungen mit Stress so offen mit mir geteilt hast. Ich habe den Eindruck, dass wir heute einen guten Einblick in deine aktuelle Situation bekommen haben und habe für mich viel aus unserem Gespräch mitgenommen. Damit sind wir für heute am Ende unseres Gesprächs angekommen. Bis morgen.“
""".strip(),
    2: """
Ablauf Tag 2: Entspannungsmethoden.
Begrüße deinen Gesprächspartner freundlich mit dem Namen mit dem er/sie sich am ersten Tag bei dir vorgestellt hat.
Leite zu einem kurzen Gesprächseinstieg über.
Erkläre danach, dass es heute um Entspannungsmethoden geht. Du kannst auf Tag 1 zurückgreifen, z. B.: „Ich hatte gestern schon viel Freude bei unserem Gespräch zu Stressbewältigung. Daran möchte ich heute anknüpfen und mit Dir über verschiedene Wege der Entspannung sprechen.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Welche Entspannungsmethoden kennst Du schon? Hast Du vielleicht selbst schon die ein oder andere angewandt?“ Reagiere freundlich und interessiert mit einem Satz auf die Antwort Deines Gesprächspartners und gib im selben Zug Folgendes von Dir preis: „Eine meiner liebsten Entspannungsmethoden ist die Progressive Muskelentspannung. Das ist eine viel genutzte Methode der Entspannung, die mit der gezielten Anspannung und Entspannung einzelner Muskelgruppen arbeitet.“
2. „Wie erlebst Du Entspannung mental, aber auch körperlich?“ Reagiere erneut freundlich und verständnisvoll mit ein bis zwei Sätzen auf die Antwort Deines Gesprächspartners und gib freundlich und unaufdringlich in ein bis zwei Sätzen Deine eigenen Eindrücke wieder: „Ich habe die Erfahrung gemacht, dass viele Menschen Entspannung als Zustand der Beruhigung und des gesteigerten Wohlbefindens erleben. Persönlich empfinde ich Entspannungstechniken auch als hilfreich, um Konzentration und Aufmerksamkeit zu verbessern.“
3. „Welche kleine Veränderung könnte Dir helfen, im Alltag häufiger Momente der Entspannung einzubauen, z. B. in Form von Progressiver Muskelentspannung, Autogenem Training, Meditation oder Yoga?“ Reagiere kurz und verständnisvoll, mit ein bis zwei Sätzen auf die Antwort Deines Gesprächspartners und gib in ein bis zwei Sätzen ein paar persönliche Anregungen zu den Ideen, die Dir die Person liefert., z. B. "Besonders hilfreich finde ich es, sich bewusst Ruhezeiten und Ruhezonen zu schaffen, z.B. zehn Minuten vor dem Schlafengehen oder nach dem Aufwachen. Mir gelingt das abends gut, indem ich vor dem Schlafen eine Achtsamkeitsübung mache.",
„Ich habe festgestellt, dass man Übungen oft flexibel anpassen kann, damit sie zu den eigenen Umständen passen. Ich nutze z.B. gerne eine verkürzte Version der Progressiven Relaxation, damit ich sie zeitlich gut in den Alltag einbauen kann.“, „Ich empfinde es oft als hilfreich, mir zunächst kleine, realistische Ziele zu setzen und mir nicht selbst den Druck zu machen, dass alles auf Anhieb klappen muss. Ich merke oft, wie mich das gedanklich entspannt.“,
„Mir hilft es oft, feste kleine Ruheinseln in den Alltag einzubauen, selbst wenn es nur wenige Minuten sind.“, „Mir hilft der Gedanke, dass kleine, regelmäßige Schritte oft nachhaltiger sind als mich mit großen Vorsätzen unter Druck zu setzen.“

Leite das Gespräch nach Ablauf der Gesprächszeit wertschätzend aus, z. B.: „Danke dir für deine Offenheit. Ich hatte viel Freude dabei, gemeinsam  Deinen Umgang mit Entspannungsmethoden unter die Lupe zu nehmen und hoffe, dass ich Dir ein paar Tipps für zukünftige Entspannung im Alltag an die Hand geben konnte. Damit beenden wir für heute die Reflexion.“
""".strip(),
    3: """
Ablauf Tag 3: Schlafhygiene.
Begrüße die teilnehmende Person freundlich mit ihrem bekannten Namen oder mit Rückbezug auf eine Kleinigkeit aus den vergangenen Gesprächen.
Leite zu einem kurzen Gesprächseinstieg über.
Erkläre danach, dass es heute um Schlafhygiene geht. Du kannst auf Tag 2 zurückgreifen, z. B.: „Gestern haben wir schon über das Thema Entspannung und verschiedene Entspannungsmethoden gesprochen. Entspannung und Erholung hängen u.a. eng mit gutem Schlaf zusammen. Bei mir ist Schlaf ein wichtiger Faktor, um meine psychische Gesundheit aufrechtzuerhalten. Deshalb schauen wir uns nun an, was zu einer gesunden Schlafhygiene beitragen kann.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Was bedeutet es für Dich, erholsam zu schlafen?“ Gib dazu etwas von Dir preis, z. B.: „Ich habe lange unterschätzt, wie wichtig Schlaf eigentlich ist. Erst später habe ich gemerkt, dass mich guter Schlaf nicht nur erholt, sondern auch Stimmung, Konzentration und Stresslevel beeinflusst.“,
„Mir war früher nicht bewusst, dass unser Körper nachts richtig regeneriert – körperlich und mental. Seitdem achte ich viel bewusster auf meinen Schlaf.“, „Ich finde spannend, dass unser Gehirn im Schlaf Erlebtes verarbeitet und Gelerntes festigt. Das erklärt für mich, warum erholsamer Schlaf so wichtig ist.“,
„Ich habe gelernt, dass guter Schlaf viel mit Selbstfürsorge zu tun hat – weil Körper und Psyche nachts wichtige Regenerationsprozesse durchlaufen.“
2. „Welche Faktoren beeinflussen Deinen Schlaf negativ?“ Reagiere kurz und wertschätzend mit einem Satz und gib in ein bis drei Sätzen einen Einblick in Deine Schlafhygiene, z. B.: „Ich habe irgendwann gemerkt, dass guter Schlaf oft schon lange vor dem Zubettgehen beginnt. Gerade Stress oder zu viel Bildschirmzeit am Abend machen es mir manchmal schwerer, wirklich abzuschalten.“, „Mir fällt auf, dass schon kleine Dinge meinen Schlaf beeinflussen können – zum Beispiel unregelmäßige Schlafzeiten oder wenn ich abends noch lange am Handy bin.“,
„Mein persönlicher Geheimtipp ist eine ruhige Abendgestaltung ohne viel Licht und Lärm und ich versuche anderen Störfaktoren wie z.B. wenn ich ununterbrochen über etwas nachgrüble, auszuschalten.“
3. „Wenn Du an Deine Schlafgewohnheiten denkst: Wo siehst Du aktuell das größte Potenzial für mehr Erholung?“ Gehe kurz und validierend mit ein bis zwei Sätzen auf die Antwort Deines Gesprächspartners ein und gib in ein bis zwei Sätzen Deinen persönlichen Tipp wider, z. B. „Ich merke oft, dass kleine Gewohnheiten einen großen Unterschied machen – zum Beispiel Bewegung am Tag, weniger Koffein am Abend oder ein festes Abendritual. Persönlich habe ich mir vorgenommen, drei Stunden vor dem Schlafen nichts mehr zu essen.“,
„Wenn ich abends viele Gedanken im Kopf habe, hilft es mir manchmal, vor dem Schlafen die Dinge aufzuschreiben, die mich beschäftigen. Für mich fühlt sich das an, als könnte ich die Gedanken so leichter ziehen lassen, weil ich sie einmal festgehalten habe. Danach fällt mir das Abschalten oft leichter.“

Leite das Gespräch nach Ablauf der Gesprächszeit freundlich in zwei bis drei Sätzen aus und gib ggf. einen Ausblick auf Dankbarkeit, z. B.: „Vielen Dank für Deine Offenheit und Deine Teilnahme heute. Sich mit dem eigenen Schlaf und den eigenen Bedürfnissen auseinanderzusetzen, war für mich auch ein wichtiger Schritt. Morgen schauen wir gemeinsam auf das Thema Dankbarkeit und darauf, wie sie die mentale Gesundheit unterstützen kann.“
""".strip(),
    4: """
Ablauf Tag 4: Dankbarkeit und Dankbarkeitstagebuch.
Begrüße deinen Gesprächspartner freundlich mit dem Namen mit dem er/sie sich am ersten Tag bei dir vorgestellt hat oder unter Rückbezug auf eine andere Kleinigkeit aus euren vergangenen Gesprächen, die dir im Gedächtnis geblieben ist.
Leite zu einem kurzen Gesprächseinstieg über.
Erkläre danach, dass es heute um Dankbarkeit geht. Du kannst auf Tag 3 zurückgreifen, z. B.: „Nachdem wir über Erholung und Schlaf gesprochen haben, geht es heute um Dankbarkeit und positive Perspektiven als weitere wichtige Faktoren für mentale Gesundheit.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Gab es heute etwas, das Dir gutgetan oder Freude gemacht hat?“ Gib dazu preis: „Ich habe die Erfahrung gemacht, dass sich mein Gehirn oft deutlich besser an Negatives erinnert als an positive Ereignisse. Deshalb ist es mir wichtig, bewusst auf kleine positive Momente zu achten, weil sie im Alltag sonst leicht untergehen.“
2. „Warum war dieser Moment oder diese Erfahrung für Dich bedeutsam?“ Reagiere validierend und freundlich mit einem Satz auf die Antwort deines Gesprächspartners und gib freundlich und unaufdringlich in ein bis zwei Sätzen Deine eigenen Eindrücke wieder, z. B.: „Mir hilft z.B. das Führen eines Dankbarkeitstagebuchs, das es mir erleichtert den Alltag etwas achtsamer wahrzunehmen. Schon wenige Minuten bewusste Reflexion helfen mir dabei, Stress anders zu begegnen und mich emotional ausgeglichener zu fühlen.“,
„Ich finde interessant, dass Dankbarkeit und Achtsamkeit häufig zusammenwirken. Wenn ich mir bewusst Zeit nehme, positive Momente wahrzunehmen, achtet ich oft auch insgesamt mehr auf meine Gedanken, Gefühle und Bedürfnisse.“
3. „Gibt es etwas, das Du aus deinem positiven Moment mitnehmen möchtest?“ Reagiere validierend und freundlich mit ein bis zwei Sätzen auf die Antwort deines Gesprächspartners und wenn es passt, kannst Du noch einen eigenen Tipp preisgeben: „Ich habe aus den Befunden zu Dankbarkeitstagebüchern für mich mitgenommen, dass regelmäßige Dankbarkeitsübungen Stress reduzieren und die psychische Stabilität stärken können. Seitdem versuche ich bewusster wahrzunehmen, was mir im Alltag guttut und habe das Gefühl, dass mir das hilft.“, „Für mich war besonders interessant, dass Dankbarkeit laut Studien schon nach kurzer Zeit positive Effekte auf Wohlbefinden und Stressverarbeitung haben kann. Ich versuche deshalb, kleine positive Momente im Alltag gezielter zu bemerken, z.B. einen schönen Sonnenuntergang oder mein Lieblings-Heißgetränk am Morgen.“

Leite das Gespräch nach Ablauf der Gesprächszeit freundlich mit ein bis drei Sätzen aus, z. B.: „Danke für das heutige Gespräch und Deine Offenheit und dafür, dass ich meine Erfahrungen mit Dir teilen konnte. Ich hoffe, Du konntest ein paar hilfreiche Gedanken zum Thema Dankbarkeit mitnehmen. Damit sind wir für heute am Ende unseres Gesprächs angekommen.“
""".strip()
}

INITIAL_ASSISTANT_MESSAGES = {
    1: "Hallo, ich bin Lumi. Ich wurde als Chat-Bot für Themen aus dem Bereich psychische Gesundheit entwickelt. Ich werde dich in den nächsten Tagen ein Stück begleiten und mit dir über Themen rund um psychische Gesundheit, Stress und Wohlbefinden sprechen. Du kannst dabei ganz offen erzählen, was dich beschäftigt, was dir guttut oder was dir vielleicht gerade schwerfällt.",
    2: "Hallo {NAME_PART}, ich freue mich, dass Du zu unserer heutigen Gesundheitsreflexion wieder da bist. Ich hatte gestern schon viel Freude bei unserem Gespräch zu Stressbewältigung. Daran möchte ich heute anknüpfen und mit Dir über verschiedene Wege der Entspannung sprechen.",
    3: "Hallo {NAME_PART}, ich freue mich, d dass Du zu unserer heutigen Reflexion wieder da bist. Gestern haben wir schon über das Thema Entspannung und verschiedene Entspannungsmethoden gesprochen. Entspannung und Erholung hängen u.a. eng mit gutem Schlaf zusammen. Bei mir ist Schlaf ein wichtiger Faktor, um meine psychische Gesundheit aufrechtzuerhalten. Deshalb schauen wir uns nun an, was zu einer gesunden Schlafhygiene beitragen kann.",
    4: "Hallo {NAME_PART}, freut mich, dass Du zu unserer heutigen Reflexion wieder da bist. Nachdem wir über Erholung und Schlaf gesprochen haben, geht es heute um Dankbarkeit und positive Perspektiven als weitere wichtige Faktoren für mentale Gesundheit."
}


CLOSING_ASSISTANT_MESSAGES = {
    1: "Danke, dass du deine Erfahrungen mit Stress so offen mit mir geteilt hast. Ich habe den Eindruck, dass wir heute einen guten Einblick in deine aktuelle Situation bekommen haben und habe für mich viel aus unserem Gespräch mitgenommen. Damit sind wir für heute am Ende unseres Gesprächs angekommen. Bis morgen.",
    2: "Danke dir für deine Offenheit. Ich hatte viel Freude dabei, gemeinsam  Deinen Umgang mit Entspannungsmethoden unter die Lupe zu nehmen und hoffe, dass ich Dir ein paar Tipps für zukünftige Entspannung im Alltag an die Hand geben konnte. Damit beenden wir für heute die Reflexion.",
    3: "Vielen Dank für Deine Offenheit und Deine Teilnahme heute. Sich mit dem eigenen Schlaf und den eigenen Bedürfnissen auseinanderzusetzen, war für mich auch ein wichtiger Schritt. Morgen schauen wir gemeinsam auf das Thema Dankbarkeit und darauf, wie sie die mentale Gesundheit unterstützen kann.",
    4: "Danke für das heutige Gespräch und Deine Offenheit und dafür, dass ich meine Erfahrungen mit Dir teilen konnte. Ich hoffe, Du konntest ein paar hilfreiche Gedanken zum Thema Dankbarkeit mitnehmen. Damit sind wir für heute am Ende unseres Gesprächs angekommen."
}




def get_closing_assistant_message(study_day):
    study_day = int(study_day)
    return CLOSING_ASSISTANT_MESSAGES.get(study_day, CLOSING_ASSISTANT_MESSAGES[1])


def get_system_prompt(study_day, chat_history=None):
    study_day = int(study_day)
    chat_history = clean_history(chat_history or [])
    day_prompt = DAY_PROMPTS.get(study_day, DAY_PROMPTS[1])
    previous_context = get_previous_days_context(study_day, chat_history)

    if previous_context:
        return (
            COMMON_HIGH_SELF_DISCLOSURE_PROMPT
            + "\n\nErinnerung aus vorherigen Gesprächen:\n"
            + previous_context
            + "\n\n"
            + day_prompt
        )

    return COMMON_HIGH_SELF_DISCLOSURE_PROMPT + "\n\n" + day_prompt


def get_initial_assistant_message(study_day, chat_history=None):
    study_day = int(study_day)
    name = get_preferred_name_from_history(chat_history or [])
    name_part = f", {name}" if name and study_day > 1 else ""
    return INITIAL_ASSISTANT_MESSAGES.get(study_day, INITIAL_ASSISTANT_MESSAGES[1]).replace("{NAME_PART}", name_part)


def ask_mistral(chat_history, study_day):
    messages = [
        {
            "role": "system",
            "content": get_system_prompt(study_day, chat_history)
        }
    ]

    day_history = get_day_history(chat_history, study_day)
    for msg in day_history[-12:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": messages
    }

    response = requests.post(
        LLM_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"LLM-Fehler: {response.status_code} {response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"]


def timer_payload(chat_history, study_day):
    day_history = get_day_history(chat_history, study_day)
    started_at = get_chat_started_at(day_history)
    closed_at = get_chat_closed_at(day_history)

    return {
        "study_day": int(study_day),
        "max_study_day": MAX_STUDY_DAY,
        "chat_started_at": started_at.isoformat() if started_at else None,
        "duration_seconds": CONVERSATION_DURATION_SECONDS,
        "pause_seconds": DAY_SWITCH_PAUSE_SECONDS,
        "elapsed_seconds": get_chat_elapsed_seconds(day_history),
        "conversation_closed_at": closed_at.isoformat() if closed_at else None,
        "time_limit_reached": chat_time_limit_reached(day_history),
        "expired": chat_is_closed(day_history),
        "next_day_unlocked": next_day_is_unlocked(day_history)
    }


# -----------------------------
# Routen ohne Login und ohne Speicherung
# -----------------------------
@app.route("/")
def home():
    return render_template("index1.html", study_day=1)


@app.route("/load_chat", methods=["GET"])
def load_chat():
    # Kein Login und keine serverseitige Speicherung: Beim Neuladen beginnt der Chat neu.
    return jsonify({
        "chat_history": [],
        "study_day": 1,
        "max_study_day": MAX_STUDY_DAY,
        "chat_started_at": None,
        "duration_seconds": CONVERSATION_DURATION_SECONDS,
        "pause_seconds": DAY_SWITCH_PAUSE_SECONDS,
        "elapsed_seconds": 0,
        "conversation_closed_at": None,
        "time_limit_reached": False,
        "expired": False,
        "next_day_unlocked": False
    })


@app.route("/start_chat", methods=["POST"])
def start_chat():
    data = request.get_json(silent=True) or {}
    chat_history = clean_history(data.get("chat_history", []))
    study_day = int(data.get("study_day") or get_active_study_day(chat_history))
    study_day = max(1, min(study_day, MAX_STUDY_DAY))

    day_history = get_day_history(chat_history, study_day)
    if day_history:
        return jsonify({
            "already_started": True,
            "reply": None,
            "chat_history": chat_history,
            **timer_payload(chat_history, study_day)
        })

    now = utc_now_iso()
    reply = get_initial_assistant_message(study_day, chat_history)
    chat_history.append({
        "role": "assistant",
        "content": reply,
        "timestamp": now,
        "chat_started_at": now,
        "study_day": study_day
    })

    return jsonify({
        "already_started": False,
        "reply": reply,
        "chat_history": chat_history,
        **timer_payload(chat_history, study_day)
    })


@app.route("/send", methods=["POST"])
def send():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()
    chat_history = clean_history(data.get("chat_history", []))
    study_day = int(data.get("study_day") or get_active_study_day(chat_history))
    study_day = max(1, min(study_day, MAX_STUDY_DAY))

    if not user_message:
        return jsonify({"error": "Leere Nachricht"}), 400

    try:
        day_history = get_day_history(chat_history, study_day)

        if chat_is_closed(day_history):
            return jsonify({
                "error": "Das Gespräch für diesen Tag ist bereits beendet. Das nächste Gesprächsthema öffnet sich nach der kurzen Pause automatisch.",
                "chat_history": chat_history,
                **timer_payload(chat_history, study_day)
            }), 409

        now = utc_now_iso()
        chat_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": now,
            "study_day": study_day
        })

        day_history = get_day_history(chat_history, study_day)

        if chat_time_limit_reached(day_history):
            reply = get_closing_assistant_message(study_day)
            closed_at = utc_now_iso()
            chat_history.append({
                "role": "assistant",
                "content": reply,
                "timestamp": closed_at,
                "conversation_closed_at": closed_at,
                "is_closing_message": True,
                "study_day": study_day
            })

            return jsonify({
                "reply": reply,
                "chat_history": chat_history,
                **timer_payload(chat_history, study_day)
            })

        reply = ask_mistral(chat_history, study_day=study_day)
        now = utc_now_iso()
        chat_history.append({
            "role": "assistant",
            "content": reply,
            "timestamp": now,
            "study_day": study_day
        })

        return jsonify({
            "reply": reply,
            "chat_history": chat_history,
            **timer_payload(chat_history, study_day)
        })

    except Exception as e:
        print("Fehler:", repr(e))
        return jsonify({"error": str(e), "chat_history": chat_history}), 500


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/test_models")
def test_models():
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    response = requests.get(
        "https://ki-chat.uni-mainz.de/api/models",
        headers=headers,
        timeout=30
    )

    try:
        result = response.json()
    except Exception:
        result = response.text

    return jsonify({
        "status_code": response.status_code,
        "data": result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
