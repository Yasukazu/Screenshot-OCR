from peewee import *

database = SqliteDatabase('/home/yasukazu/Documents/screen/2025/txt_lines.sqlite')

class UnknownField(object):
    def __init__(self, *_, **__): pass

class BaseModel(Model):
    class Meta:
        database = database

class TextLines03(BaseModel):
    app = IntegerField(null=True)
    day = IntegerField(null=True)
    stem = TextField(null=True)
    title = TextField(null=True)
    txt_lines = BlobField(null=True)
    wages = IntegerField(null=True)

    class Meta:
        table_name = 'text_lines-03'
        indexes = (
            (('app', 'day'), True),
        )
        primary_key = CompositeKey('app', 'day')

class TextLines04(BaseModel):
    app = IntegerField(null=True)
    day = IntegerField(null=True)
    stem = TextField(null=True)
    title = TextField(null=True)
    txt_lines = BareField(null=True)
    wages = IntegerField(null=True)

    class Meta:
        table_name = 'text_lines-04'
        indexes = (
            (('app', 'day'), True),
        )
        primary_key = CompositeKey('app', 'day')

class TxtLines05(BaseModel):
    app = IntegerField(null=True)
    checksum = TextField(null=True, unique=True)
    day = IntegerField(null=True)
    stem = TextField(null=True, unique=True)
    title = TextField(null=True)
    txt_lines = BlobField(null=True)
    wages = IntegerField(null=True)

    class Meta:
        table_name = 'txt_lines-05'
        indexes = (
            (('app', 'day'), True),
        )
        primary_key = CompositeKey('app', 'day')

class TxtLines08(BaseModel):
    app = IntegerField(null=True)
    checksum = TextField(null=True, unique=True)
    day = IntegerField(null=True)
    stem = TextField(null=True, unique=True)
    title = TextField(null=True)
    txt_lines = BlobField(null=True)
    wages = IntegerField(null=True)

    class Meta:
        table_name = 'txt_lines-08'
        indexes = (
            (('app', 'day'), True),
        )
        primary_key = CompositeKey('app', 'day')

class TxtLines09(BaseModel):
    app = IntegerField(null=True)
    checksum = TextField(null=True, unique=True)
    day = IntegerField(null=True)
    stem = TextField(null=True, unique=True)
    title = TextField(null=True)
    txt_lines = BlobField(null=True)
    wages = IntegerField(null=True)

    class Meta:
        table_name = 'txt_lines-09'
        indexes = (
            (('app', 'day'), True),
        )
        primary_key = CompositeKey('app', 'day')

