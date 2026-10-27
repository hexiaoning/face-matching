"""Explicit gallery constraints derived from validated citizen identity numbers."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import re


def china_today() -> date:
    return datetime.now(timezone(timedelta(hours=8))).date()


@dataclass(frozen=True)
class IdentityDetails:
    sex: str
    birth_date: date
    region_code: str

    def age_on(self, today: date) -> int:
        return today.year - self.birth_date.year - ((today.month, today.day) <
                                                   (self.birth_date.month, self.birth_date.day))


def parse_identity(value: str, today: date | None = None) -> IdentityDetails | None:
    value = value.strip().upper()
    if not re.fullmatch(r"[1-9][0-9]{16}[0-9X]", value):
        return None
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    if "10X98765432"[sum(int(a) * b for a, b in zip(value[:17], weights)) % 11] != value[-1]:
        return None
    try:
        birth = datetime.strptime(value[6:14], "%Y%m%d").date()
    except ValueError:
        return None
    if birth > (today or china_today()) or value[14:17] == "000":
        return None
    return IdentityDetails("男" if int(value[16]) % 2 else "女", birth, value[:6])


@dataclass(frozen=True)
class PersonFilter:
    library_id: str | None = None
    sex: str | None = None
    region_prefix: str = ""
    min_age: int | None = None
    max_age: int | None = None

    def __post_init__(self) -> None:
        if self.sex not in (None, "男", "女"):
            raise ValueError("性别须为男、女或不限")
        if self.region_prefix and not re.fullmatch(r"(?:[0-9]{2}|[0-9]{4}|[0-9]{6})", self.region_prefix):
            raise ValueError("地区码须为 2 位省、4 位市或 6 位区县代码")
        if any(age is not None and not 0 <= age <= 150 for age in (self.min_age, self.max_age)):
            raise ValueError("年龄范围须在 0～150 岁")
        if self.min_age is not None and self.max_age is not None and self.min_age > self.max_age:
            raise ValueError("最小年龄不能大于最大年龄")

    def matches(self, id_card: str, library_ids: tuple[str, ...], today: date | None = None) -> bool:
        if self.library_id is not None and self.library_id not in library_ids:
            return False
        if self.sex is None and not self.region_prefix and self.min_age is None and self.max_age is None:
            return True
        today = today or china_today()
        details = parse_identity(id_card, today)
        if details is None:
            return False
        age = details.age_on(today)
        return ((self.sex is None or details.sex == self.sex)
                and details.region_code.startswith(self.region_prefix)
                and (self.min_age is None or age >= self.min_age)
                and (self.max_age is None or age <= self.max_age))
