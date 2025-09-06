# app/db/base.py
-from sqlmodel import SQLModel, create_engine, Session
-from app.config import settings
-import os
+from sqlmodel import SQLModel, create_engine, Session
+from app.config import settings
+import os

 # Ensure data dir exists (for SQLite)
 if settings.db_dsn.startswith("sqlite"):
     os.makedirs("data", exist_ok=True)

 engine = create_engine(settings.db_dsn, echo=False)

 def init_db():
     from app.db import models  # noqa: F401 ensure models register
     SQLModel.metadata.create_all(engine)

-class get_session:
-    def __call__(self):
-        with Session(engine) as session:
-            yield session
+def get_session():
+    with Session(engine) as session:
+        yield session
