# Note these exercises involve installation of 
# docker
# edgedb client and server

import edgedb
from datetime import datetime

edgedb.connect('edgedb://edgedb@localhost/ambv')
conn=_
conn.execute("""
CREATE TYPE User {
    CREATE REQUIRED PROPERTY name -> str;
    CREATE PROPERTY date_of_birth -> cal::local_date;
}
""")

conn.query("""
INSERT User {
    name :=<str>$name,
    date_of_birth := <cal::local_date>$dob
}
""", name = 'R Kashi', dob=datetime.date(1969, 5, 13))

