"""Food-safety verdict: is produce from the scanned crop safe to eat?

The verdict comes from the recognised condition's knowledge-base entry, adjusted
for things a photo cannot show (recent pesticide spraying) and for uncertain
predictions. It only covers what is visible; it is guidance, not a lab test.
"""

VERDICTS = {
    "safe": {"title": "Safe to eat", "short": "Safe", "icon": "shield"},
    "caution": {"title": "Eat with care", "short": "Caution", "icon": "alert"},
    "unsafe": {"title": "Not safe to eat", "short": "Not safe", "icon": "x"},
    "unknown": {"title": "Can't tell from this photo", "short": "Unknown", "icon": "help"},
}
RANK = {"safe": 0, "caution": 1, "unsafe": 2}

ALWAYS = ("This check is based only on what is visible in the photo. It cannot detect pesticide "
          "residues or invisible toxins. When in doubt, throw it out.")


def assess(info: dict, uncertain: bool, sprayed: str = "no", is_demo: bool = False) -> dict:
    """Return {status, title, reason, advice[], note} for the result page and API.

    sprayed: "no" | "yes" (sprayed in the last 2 weeks) | "unsure"
    """
    if uncertain:
        status = "unknown"
        reason = ("The model is not confident about this image. The crop or condition may not be one it "
                  "was trained on, or the photo may be unclear.")
        advice = ["Retake the photo closer, in daylight, with the damaged part in focus",
                  "Do not eat produce that is mouldy, rotting, slimy or smells bad",
                  "Show the crop to a MoFA extension officer if unsure"]
    else:
        ed = info.get("edibility") or {"status": "unknown", "reason": "No food-safety data for this condition.",
                                       "advice": []}
        status, reason, advice = ed["status"], ed["reason"], list(ed["advice"])

    if sprayed == "yes" and status in RANK:
        if status == "safe":
            status = "caution"
        reason += " The crop was recently sprayed, so pesticide residue is possible."
        advice.insert(0, "Wait for the pre-harvest interval (PHI) on the product label before harvesting or "
                         "eating; wash produce thoroughly")
    elif sprayed == "unsure" and status in RANK:
        advice.append("If the crop may have been sprayed, check the product label's pre-harvest interval "
                      "before eating")

    verdict = VERDICTS.get(status, VERDICTS["unknown"])
    return {"status": status, "title": verdict["title"], "short": verdict["short"],
            "reason": reason, "advice": advice, "note": ALWAYS, "sprayed": sprayed,
            "simulated": is_demo}
