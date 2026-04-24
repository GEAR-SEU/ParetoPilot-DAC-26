import concurrent.futures
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config_loader import get_config_value


@dataclass
class TokenStats:
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    total_tokens: list[int] = field(default_factory=list)

    def record(self, usage):
        self.prompt_tokens.append(usage.get("prompt_tokens", 0))
        self.completion_tokens.append(usage.get("completion_tokens", 0))
        self.total_tokens.append(usage.get("total_tokens", 0))


class BaseDatasetProcessor:
    log_filename = "dataset.log"
    max_workers = 20
    input_extension = ".tcl"
    project_names: list[str] = []

    def __init__(self):
        logging.basicConfig(
            filename=self.log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.chat = self.create_chat_model()
        self.stats = TokenStats()

    @staticmethod
    def create_chat_model():
        return ChatOpenAI(
            api_key=get_config_value("llm", "api_key", default="") or None,
            base_url=get_config_value("llm", "api_base", default="") or None,
            model=get_config_value("llm", "model", default="gpt-4o-mini"),
            temperature=0.7,
        )

    def build_prompt(self, *args):
        raise NotImplementedError

    def get_system_prompt(self):
        raise NotImplementedError

    def get_task_directory(self, project):
        raise NotImplementedError

    def read_task_inputs(self, project, input_filename):
        raise NotImplementedError

    def discover_tasks(self):
        tasks = []
        for project in self.project_names:
            task_dir = self.get_task_directory(project)
            if not os.path.isdir(task_dir):
                self.logger.warning(f"Directory not found, skipping project: {task_dir}")
                continue

            all_files = [file_name for file_name in os.listdir(task_dir) if file_name.endswith(self.input_extension)]
            all_files.sort()
            for filename in all_files[:10]:
                tasks.append((project, filename))
        return tasks

    def process_task(self, project, input_filename):
        try:
            prompt_args = self.read_task_inputs(project, input_filename)
            prompt = self.build_prompt(*prompt_args)
            messages = [
                SystemMessage(content=self.get_system_prompt()),
                HumanMessage(content=prompt),
            ]
            response = self.chat.invoke(messages)
            token_usage = response.response_metadata.get("token_usage", {})
            self.logger.info(response.response_metadata)
            return {
                "status": "success",
                "project": project,
                "filename": input_filename,
                "token_usage": token_usage,
            }
        except FileNotFoundError as exc:
            self.logger.error(f"Input file not found for {project}/{input_filename}: {exc}")
            return {"status": "error", "msg": str(exc)}
        except Exception as exc:
            self.logger.error(f"Processing failed for {project}/{input_filename}: {exc}")
            return {"status": "error", "msg": str(exc)}

    def log_summary(self):
        self.logger.info("\n" + "=" * 40)
        self.logger.info("TOKEN USAGE STATISTICS")
        self.logger.info("=" * 40)

        count = len(self.stats.total_tokens)
        if count > 0:
            total_p = sum(self.stats.prompt_tokens)
            total_c = sum(self.stats.completion_tokens)
            total_t = sum(self.stats.total_tokens)
            avg_p = total_p / count
            avg_c = total_c / count
            avg_t = total_t / count

            self.logger.info(f"Successful requests: {count}")
            self.logger.info("-" * 40)
            self.logger.info(f"{'Type':<20} | {'Total':<12} | {'Average':<12}")
            self.logger.info("-" * 40)
            self.logger.info(f"{'Prompt Tokens':<20} | {total_p:<12} | {avg_p:<12.2f}")
            self.logger.info(f"{'Completion Tokens':<20} | {total_c:<12} | {avg_c:<12.2f}")
            self.logger.info(f"{'Total Tokens':<20} | {total_t:<12} | {avg_t:<12.2f}")
        else:
            self.logger.info("No successful requests recorded.")

        self.logger.info("=" * 40)
        self.logger.info("All tasks completed.")

    def run(self):
        tasks = self.discover_tasks()
        self.logger.info("Discovering tasks...")

        if not tasks:
            self.logger.info("No tasks found to process. Please check the input directory structure.")
            self.log_summary()
            return

        self.logger.info(f"Found {len(tasks)} tasks. Starting parallel processing with {self.max_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_task, project, input_filename) for project, input_filename in tasks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing files"):
                try:
                    result = future.result()
                    if result["status"] == "success":
                        self.stats.record(result["token_usage"])
                except Exception as exc:
                    self.logger.error(f"A task failed with an unexpected exception: {exc}")

        self.log_summary()
