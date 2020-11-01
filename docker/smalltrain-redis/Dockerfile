FROM redis
RUN apt-get -y update
RUN apt-get -y install vim
RUN apt-get -y install procps

COPY redis.conf /usr/local/etc/redis/redis.conf

# Exec setup shell script in ENTRYPOINT
COPY entrypoint_smalltrain_redis.sh /usr/local/bin/
COPY run_smalltrain_redis.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/*.sh
ENTRYPOINT ["/usr/local/bin/entrypoint_smalltrain_redis.sh"]

