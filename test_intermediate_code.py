from model.models import *


def test_pydantic_class():
    c = ChangesFormat(Pages="Page 1", Changes="There is an extra field")
    print(c.model_dump_json())


if __name__ == __name__:
    test_pydantic_class()
