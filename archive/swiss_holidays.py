import holidays
import logging
from datetime import date
from dateutil.relativedelta import relativedelta

def get_past_three_months_holidays(canton=None):
    """
    Gibt die Feiertage der letzten drei Monate in der angegebenen Schweiz (Kanton optional) zurück.

    :param country: Ländercode (Standard: 'CH' für Schweiz)
    :param canton: Optionaler Kanton für kantonale Feiertage (z.B. 'ZH' für Zürich)
    :return: Dictionary mit Datum als Schlüssel und Feiertagsname als Wert
    """
    heute = date.today()
    drei_monate_zurueck = heute - relativedelta(months=4)
    # Feiertage abrufen
    if canton:
        # Wenn ein Kanton angegeben ist, spezifiziere diesen
        schweiz_feiertage = holidays.CH(prov=canton, years=range(drei_monate_zurueck.year, heute.year + 1))
    else:
        # Allgemeine Feiertage ohne Kantonsspezifizierung
        schweiz_feiertage = holidays.CH(years=range(drei_monate_zurueck.year, heute.year + 1))

    result = []
    for dt in schweiz_feiertage.keys():
        result.append(dt.strftime("%d.%m.%Y"))
    return result

