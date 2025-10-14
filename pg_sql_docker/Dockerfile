# PostgreSQL Dockerfile with persistence
FROM postgres:15-alpine

RUN apk add --no-cache bash

# Set environment variables
ENV POSTGRES_DB=myapp
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=password
ENV POSTGRES_HOST_AUTH_METHOD=trust
ENV PGDATA=/var/lib/postgresql/data/pgdata

# Create directory for data persistence
RUN mkdir -p /var/lib/postgresql/data

# Copy initialization scripts (optional)
# COPY init-scripts/ /docker-entrypoint-initdb.d/

# Create a volume for data persistence
VOLUME ["/var/lib/postgresql/data"]

# Expose PostgreSQL port
EXPOSE 5432

CMD ["postgres"]
# The CMD is inherited from the base postgres image