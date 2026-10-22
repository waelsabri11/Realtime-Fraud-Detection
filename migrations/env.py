from alembic import context
from sqlalchemy import create_engine

from fraudrt.config import get_settings
from fraudrt.db.models import Base

target_metadata = Base.metadata


def run_migrations_online() -> None:
    url = context.config.get_main_option("sqlalchemy.url") or get_settings().database_url
    engine = create_engine(url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
