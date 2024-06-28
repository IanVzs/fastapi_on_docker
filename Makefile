export GIT_COMMIT := $(shell git log -1 --format=%h)
export BUILD_DATE := $(shell date +'%Y%m%d')

run_dev:
	docker run -it -p 8099:80 \
	-e MYSQL_CRUD_HOST=localhost \
	-e MYSQL_CRUD_PORT=3306 \
	-e MYSQL_CRUD_DB=crud \
	-e MYSQL_CRUD_USER=root \
	-e MYSQL_CRUD_PASSWD=testpasswd \
	--name app app-$(GIT_COMMIT)-$(BUILD_DATE):latest

build_dev:
	docker build -t app-$(GIT_COMMIT)-$(BUILD_DATE):latest .

run:
	cd app && export MYSQL_CRUD_HOST=localhost && export MYSQL_CRUD_PORT=3316 && export MYSQL_CRUD_DB=crud && export MYSQL_CRUD_USER=root && export MYSQL_CRUD_PASSWD=testpasswd && python main.py