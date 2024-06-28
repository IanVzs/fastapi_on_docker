mysqldump -h{host} -u{user} -p{passwd} crud crud_app > crud_app.sql
sed -i 's/crud_app/crud_app_current/g' crud_app.sql 
mysql -u{user} -p{passwd} crud < crud_app.sql
