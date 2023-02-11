FROM elixir:alpine

COPY . .

RUN rm -Rf _build && \
    mix local.hex --force && \
    mix local.rebar --force && \
    mix deps.get && \
    mix predict