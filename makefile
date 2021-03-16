PACKAGE_NAME=typodetect
IMAGE_NAME=docker.oa.com/nlpc/typodetect:1.2
CONTAINER_NAME=typodetect
PACKAGE_NAME=typodetect
TOOLS_DIR=tool
PROTO_FILES=$(wildcard $(PACKAGE_NAME)/proto/*.proto)
PROTO_OUT_FILES=$(PROTO_FILES:${PACKAGE_NAME}/proto/%.proto=$(PACKAGE_NAME)/proto/%_pb2.py)
IMAGE=docker.oa.com/nlpc/typodetect_data:0.2.0
RUN_DOCKER=docker run --log-driver json-file --net=host -v $(PWD):$(PWD) -u $(shell id -u):$(shell id -g) -w $(PWD) --rm
GRPC_TOOL=$(RUN_DOCKER) $(IMAGE) /usr/local/bin/python3  -m grpc_tools.protoc -I. --python_out=.  --grpc_python_out=.
PEP8_TOOL=$(RUN_DOCKER) $(IMAGE) flake8
ISORT_TOOL=$(RUN_DOCKER $(IMAGE) isort
ALL_PY_FILES=$(shell find ${PACKAGE_NAME} -name '*.py' -type f)

.PHONY: compile_proto setup start kill restart build_wheel build_image help pep8 isort

help:
	@echo "支持下面的make target"
	@echo "  compile_proto  编译proto文件"
	@echo "  setup          在当前目录下初始化空项目"
	@echo "  start          启动服务"
	@echo "  kill           停止服务"
	@echo "  test           测试grpc协议"
	@echo "  http_test      测试http+json协议"
	@echo "  restart        start + kill"
	@echo "  build_wheel    编译代码为python wheel, 输出到当前目录下的build/"
	@echo "  build_image    打包代码为docker image, 名字见makefile中的CONTAINER_NAME"
	@echo "  show           打印正在执行的服务容器中的进程"
	@echo "  sh             启动容器内的shell"
	@echo "  pep8           pep8语法检查"
	@echo "  isort          对代码中的import排序"

${PACKAGE_NAME}/proto/__init__.py: ${PROTO_FILES}
	touch $@

compile_proto: ${PACKAGE_NAME}/proto/__init__.py

setup:
	@$(TOOLS_DIR)/setup_example_project.sh $(PACKAGE_NAME)

$(PACKAGE_NAME)/proto/%_pb2.py: $(PACKAGE_NAME)/proto/%.proto
	$(GRPC_TOOL) $<


build/.lock: ${PROTO_OUT_FILES} ${ALL_PY_FILES} setup.py
	mkdir -p build
	${RUN_DOCKER} ${IMAGE} pip3 wheel -w build --no-deps .
	touch $@

build_wheel: build/.lock pep8 isort

build_image: build_wheel
	docker build --network=host -t ${IMAGE_NAME} -f docker/Dockerfile .

pep8: build/.pep8_lock
build/.pep8_lock: ${ALL_PY_FILES}
	${PEP8_TOOL} ${PACKAGE_NAME} --exclude ${PACKAGE_NAME}/utils/,${PACKAGE_NAME}/tegmonitor/,${PACKAGE_NAME}/proto/,${PACKAGE_NAME}/include/,${PACKAGE_NAME}/client/
	touch $@

isort: build/.isort_lock
build/.isort_lock: ${ALL_PY_FILES}
	@find . -type f -name '*.py' -not -path "./${PACKAGE_NAME}/proto/*" | xargs ${ISORT_TOOL}
	@touch $@
clean:
	rm -rf build/src
# vim:set filetype=make:

GRPC_PORT=9556
HTTP_PORT=9558

start: build_image
	@mkdir -p log
	if [[ -z "$$(docker ps -q --filter name=${CONTAINER_NAME})" ]]; then\
         docker run -d --net=host --rm  --name=$(CONTAINER_NAME) -it \
		 	--cap-add=SYS_PTRACE \
			-v ${PWD}/log:/app/log  \
		    -v /usr/local/zhiyan/:/usr/local/zhiyan/ \
			-v /usr/local/services/AttaAgent-1.7/:/usr/local/services/AttaAgent-1.7/ \
            -e GRPC_PORT=${GRPC_PORT} -e HTTP_PORT=${HTTP_PORT} \
			-e IMAGE=${IMAGE_NAME} \
			-e GRACE_PERIOD=60 -e WAIT_PERIOD=60 \
			-e RELOAD_CYCLE=600 -e PROCESS_NUM=12 \
			-e INTERNAL_PROCESS_NUM=30 \
	        -e THREAD_NUM=1 -e GRPC_THREAD_NUM=1\
			$(IMAGE_NAME); fi

kill:
	@while [[ ! -z "$$(docker ps -q  --filter name=${CONTAINER_NAME})" ]]; do docker kill ${PACKAGE_NAME} 2>&1 >/dev/null; done
	@while [[ ! -z "$$(docker ps -q -a --filter name=${CONTAINER_NAME})" ]]; do docker rm -f ${PACKAGE_NAME} 2>&1 >/dev/null; done
	rm -rf log/*

restart: kill start

reload:
	docker exec -it ${CONTAINER_NAME} reload.sh

sh:
	docker exec -it ${CONTAINER_NAME} bash

test:
	docker exec ${CONTAINER_NAME} python3 -m ${PACKAGE_NAME}.client --addr 127.0.0.1:${GRPC_PORT} --name GetFeature --cls Request --value 1

test_http:
	curl  localhost:${HTTP_PORT}/typoserver -d '{"doc_id": "testid"}'

show:
	docker exec ${CONTAINER_NAME} ps aux

logf:
	docker logs -f ${CONTAINER_NAME}

log:
	docker logs ${CONTAINER_NAME}
