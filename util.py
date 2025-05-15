def format_soap_markdown(soap: dict) -> str:
    return f"""
**S - Subjective:** {", ".join(soap['subjective'])}
**O - Objective:** {", ".join(soap['objective'])}
**A - Assessment:** {", ".join(soap['assessment'])}
**P - Plan:** {", ".join(soap['plan'])}
"""
