# Access MongoDB shell on the container
```
$ docker exec -it $CONTAINER_ID /bin/bash
root@CONTAINER_ID:/# mongo
MongoDB server version: 4.0.9
...
>
````

Create DB
````
> use ggdb
````

# Connect to admin DB
```
> use admin
```

# Add admin user
```
> db.createUser({
      user:"admin_xxxxxxxxxx",
      pwd: "xxxxxxxxxx",
      roles:[
        { role:"userAdminAnyDatabase", db:"admin" },
        { role:"readWrite", db:"ggdb" }
       ]
    })

> db.createUser({
      user:"ggdb_xxxxxxxxxx",
      pwd: "xxxxxxxxxx",
      roles:[
        { role:"readWrite", db:"ggdb" }
       ]
    })
```


# change password
```
> db.updateUser("ggdb_xxxx", {pwd: "new_password" })
```

# check user and password
```
> use admin
> db.system.users.find()
> db.auth("ggdb_xxxx","password")
```


# Change setting to enable password authorization
```
$ vi /etc/mongod.conf
security:
  authorization: enabled
```

# Connection test
$ mongo [db_name] --host [host_name] --port [port_num] --username [username] --password [password] --authenticationDatabase [auth_db_name]
$ mongo ggdb --host localhost --port 27017 --username "ggdb_xxxx" --password "xxxxx" --authenticationDatabase admin

```
