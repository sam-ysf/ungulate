#pragma once

#include <string>

namespace util::file {
    /*! Database schema
     */
    const std::string kSchema = R"(
create table if not exists models
(
    model
        text not null,
    description
        text,

    unique(model)
);

create table if not exists files
(
    id
        integer primary key,
    model
        text not null,
    prompt
        text not null,
    filename
        text not null,
    path
        text not null,
    hash
        blob not null,
    magic
        text not null,
    n_pages
        integer not null,

    unique(model, prompt, path, hash)
);

create table if not exists file_metadata
(
    file
        integer not null,
    key
        text not null,
    value
        text not null,

    unique(file, key),

    foreign key(file) references files(id)
);

create virtual table if not exists documents using fts5
(
    file,
    page_in_file,
    text
);)";
} // namespace util::file
