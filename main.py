import asyncio
import aiohttp
import json
import os
import time
import sqlite3 as sq
import random as rnd


def response_string_to_tag_list(response: str):
    tag_list = [line for line in response.split('\n') if line != ""]
    tag_list = [line.split(' ') for line in tag_list]
    tag_list = [[word.lower() for word in line if word not in ['', '-']] for line in tag_list]
    tag_list = [" ".join(line) for line in tag_list]
    return tag_list


async def get_model_one_session_response(prompt: str, link: str, post_headers: dict, post_body: dict):
    post_body["messages"] = [{
        "role": "user",
        "content": prompt
    }]

    answer_str = ""

    async with aiohttp.ClientSession() as session:
        async with session.post(link, headers=post_headers, json=post_body) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = data.strip()
                        if chunk:
                            chunk_json = json.loads(chunk)
                            choices = chunk_json['choices']
                            content = None if choices == [] else choices[0]['delta']['content']
                            if isinstance(content, str):
                                answer_str += content
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")

    return answer_str


class ArchiveClassifier:
    link = "https://llm.chutes.ai/v1/chat/completions"

    def __init__(self, path: str):
        #
        if not os.path.exists(path):
            raise FileNotFoundError(f"Архив не найден: {path}")
        self.path = path

        #
        self.documents_folder = f"{path}/text_documents"
        if not os.path.exists(self.documents_folder):
            raise FileNotFoundError(
                "Папка должна содержать подпапку 'text_documents' со списком текстовых документов")
        self.document_names = os.listdir(self.documents_folder)

        #
        if not os.path.exists(f"{path}/post_format.json"):
            raise FileNotFoundError("Папка должна содержать файл post_format.json")

        with open(f"{path}/post_format.json", 'r', encoding='utf-8') as f:
            post_format = json.loads(f.read())

        self.post_body = post_format["body"]
        self.post_body["stream"] = True
        self.post_headers = post_format["headers"]
        self.post_headers["Content-Type"] = "application/json"

        #
        self.items = set()
        self.prompt_format = ""
        with open(f"{path}/prompt_format.txt", 'r', encoding='utf-8') as f:
            for line in f:
                if "#include_list" in line:
                    b = line.index('"')
                    e = line.index('"', b + 1)
                    with open(f"{path}/{line[b + 1:e]}", 'r', encoding='utf-8') as f:
                        items = json.loads(f.read())
                    self.prompt_format += "\n".join(items) + "\n"
                    self.items = self.items.union(items)
                elif "#include_text" in line:
                    b = line.index('"')
                    e = line.index('"', b + 1)
                    with open(f"{path}/{line[b + 1:e]}", 'r', encoding='utf-8') as f:
                        self.prompt_format += f.read()
                else:
                    self.prompt_format += line

        #
        self.result_con = sq.connect(f"{path}/result.db")
        self.result_cur = self.result_con.cursor()

        if not self.table_exists("processing", "epoch_count", "result"):
            self.result_cur.executescript("""
                create table "processing(epoch_count)"(
                    cnt integer default 0
                );
                insert into "processing(epoch_count)" (cnt) select 0;
                create table "processing(autocorrect)"(
                    wrong text not null,
                    right text not null
                );
            """)
            self.result_con.commit()

        self.test_con = sq.connect(f"{path}/test.db")
        self.test_cur = self.test_con.cursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.test_con.close()
        self.result_con.close()

    def processing_count(self):
        self.result_cur.execute("""
            select cnt from "processing(epoch_count)"
        """)
        return [*self.result_cur.fetchone()][0]

    def increment_processing_count(self):
        self.result_cur.execute("""
            update "processing(epoch_count)"
            set cnt = cnt + 1
            where rowid = 1
        """)
        self.result_con.commit()

    def get_database_cursor(self, database: str):
        database = database.lower()
        if database not in ["test", "result"]:
            raise RuntimeError("Tag must either be from test or result")
        return self.test_cur if database == "test" else self.result_cur

    def table_exists(self, type: str, table: str, database: str):
        cur = self.get_database_cursor(database)
        cur.execute(f"""
            select count(*) FROM sqlite_master
            where type='table' and name='{type}({table})';
        """)
        return cur.fetchone() != (0,)

    def generate_tag_list(self, prompt: str):
        response = asyncio.run(get_model_one_session_response(
            prompt, self.link, self.post_headers, self.post_body
        ))
        return response_string_to_tag_list(response)

    def generate_prompt(self, document: str):
        with open(f"{self.documents_folder}/{document}/text.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        return self.prompt_format.format(DOCUMENT=content)

    def load_tag(self, tag: str, database: str):
        cur = self.get_database_cursor(database)
        if not self.table_exists("tag", tag, database):
            return {}
        cur.execute(f"""
            select * from "tag({tag})"
        """)
        return dict(cur.fetchall())

    def load_document(self, document: str, database: str):
        cur = self.get_database_cursor(database)
        cur.execute("""
            select name from sqlite_master
            where type = 'table'
        """)
        tags = {}
        for tag, in cur.fetchall():
            cur.execute(f"""
                select {"counter" if database == "result" else "prob"} from "tag({tag})"
                where document = "{document}"
                limit 1
            """)
            find = cur.fetchone()
            if find:
                tags[tag], = find

        return tags

    def increment_row_count(self, tag: str, document: str):
        self.result_cur.executescript(f"""
            create table if not exists "tag({tag})"(
                document text not null,
                counter integer default 0
            );
            create index if not exists "index({tag})" on "tag({tag})"(document);
        """)
        self.result_con.commit()

        self.result_cur.execute(f"""
            select 1 from "tag({tag})"
            where document = "{document}"
            limit 1
        """)

        if self.result_cur.fetchone() is None:
            self.result_cur.execute(f"""
                insert into "tag({tag})" (document) select "{document}"
            """)
            self.result_con.commit()

        self.result_cur.execute(f"""
            update "tag({tag})"
            set counter = counter + 1
            where document = "{document}"
        """)

        self.result_con.commit()

    def __classify_once__(self):
        classification_start_time = time.time()

        require_correction = {}

        self.increment_processing_count()

        for i, document in enumerate(self.document_names):
            task_classification_start_time = time.time()

            print(f"Документ, классифицируемый в данный момент: {document}")

            prompt = self.generate_prompt(document)

            for tag in self.generate_tag_list(prompt):
                if tag in self.items:
                    self.increment_row_count(tag, document)
                else:
                    if tag in require_correction:
                        require_correction[tag].add(document)
                    else:
                        self.result_cur.execute(f"""
                            select right from "processing(autocorrect)"
                            where wrong = "{tag}"
                        """)
                        result = self.result_cur.fetchone()
                        if result is None:
                            require_correction[tag] = {document}
                        else:
                            right, = result
                            if right != "_удалить":
                                self.increment_row_count(right, document)

            percent = (i + 1) / len(self.document_names) * 100

            print(f"{i + 1}/{len(self.document_names)} документов классифицированы ({percent:.2f}%)")
            print(f"Затраченное время: {time.time() - task_classification_start_time:.3f} секунды\n")

        print("Все документы классифицированы")
        print(f"Затраченное время: {time.time() - classification_start_time: .3f} секунды\n")

        return require_correction

    def documents_from_tags(self, tags: list):
        documents = {}
        processing_count = self.processing_count()
        for tag in tags:
            self.result_cur.execute(f"""
                select document, counter from "tag({tag})"
            """)
            for document, counter in self.result_cur.fetchall():
                if document not in documents:
                    documents[document] = {"counter": 0, "prob": 1}
                documents[document]["counter"] += 1
                documents[document]["prob"] *= (counter / processing_count)
        return dict([
            (task, value["prob"])
            for task, value in documents.items()
            if value["counter"] == len(tags)
        ])

    def task_probabilities(self, document: str):
        processing_count = self.processing_count()
        result_documents = self.load_document(document, "result")
        for key in result_documents:
            result_documents[key] /= processing_count
        return result_documents

    def calculate_error(self):
        error = mass = 0

        processing_count = self.processing_count()

        for tag in self.items:
            test_documents = self.load_tag(tag, "test")
            result_documents = self.load_tag(tag, "result")

            test_keys = set() if not test_documents else {*test_documents.keys()}
            result_keys = set() if not result_documents else {*result_documents.keys()}

            for task in test_keys.union(result_keys):
                test_prob = 0 if task not in test_keys else test_documents[task]
                result_counter = 0 if task not in result_keys else result_documents[task]
                mass += test_prob
                error += abs(test_prob - result_counter / processing_count)

        return error / mass

    def classify(self, epoch_count: int):
        for epoch in range(epoch_count):
            print(f"Начало эпохи под номером {epoch + 1}/{epoch_count}\n")

            require_correction = self.__classify_once__()

            if require_correction:
                print("Определённые сгенерированные тэги отсутствуют в списке")
                print("Напишите, к каким тэгам из списка вы бы их отнесли")
                print("Если хотите исключить тэг, то напишите '_удалить'\n")

                for tag, tasks in require_correction.items():
                    if tag == "_удалить":
                        continue
                    while True:
                        received = input(f"{tag}: ")
                        if received in self.items or received == '_удалить':
                            break
                        print(f"\nВведённый тэг '{received}' не распознан. Попробуйте снова.\n")
                    self.result_cur.execute(f"""
                        insert into "processing(autocorrect)" (wrong, right) select "{tag}", "{received}"
                    """)
                    self.result_con.commit()
                    if received != '_удалить':
                        for task in tasks:
                            self.increment_row_count(received, task)

                print()

            print(f"Эпоха под номером {epoch + 1}/{epoch_count} закончилась")


def main():
    path = "D:/classified_db"
    with ArchiveClassifier(path) as ac:
        print(ac.generate_prompt("Данил и растения"))
        # ac.classify(epoch_count=1)


if __name__ == '__main__':
    main()
