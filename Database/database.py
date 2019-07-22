import MySQLdb
import datetime

def mysqlconnect():
    conn = MySQLdb.connect(host= "localhost",
                  user="root",
                  passwd="",
                  db="python_db")

    if (conn):
        # Carry out normal procedure
        print("Connection successful")
    else:
        # Terminate
        print("Connection unsuccessful")

    return conn


def mysqlInsert(object , count):
   #mysqlOgInsert(object,count)
   print(object + ":" + str(count))



def mysqlOgInsert(object , count):
    conn=mysqlconnect()
    cursor = conn.cursor()
    today = datetime.datetime.now()
    # datetime.date(datetime.now())
    currentDT = today.strftime("%Y-%m-%d %H:%M:%S")
    print(currentDT)
    cursor.execute("SELECT count FROM detection WHERE object = '%s'" % object)
    result_set = cursor.fetchall()
    value=len(result_set)
    print(value)
    if value>=1:
        print("Update")
        counter=(result_set[0])[0]
        cursor.execute("UPDATE detection SET count=%s , date=%s WHERE object=%s", ((counter+1),currentDT, object))
        row = cursor.fetchall()
    else:
       print("Insert")
       cursor.execute("INSERT INTO detection (date, object, count) VALUES (%s, %s, %s)",
                      (currentDT, object, count))
       row = cursor.fetchall()



# mysqlOgInsert("SAR",1)
# mysqlOgInsert("SAR",1)
# mysqlOgInsert("SAR",1)
# mysqlOgInsert("ABC",1)
# mysqlOgInsert("XYz",1)
# mysqlOgInsert("RST",1)
#
# mysqlOgInsert("XYz",1)
# mysqlOgInsert("RST",1)