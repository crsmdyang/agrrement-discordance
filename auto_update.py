"""
자동 업데이트 스크립트
- Git 저장소가 깨끗한지 확인한 뒤 원격 변경 사항을 가져와 fast-forward pull
- 필요한 경우 requirements.txt를 재설치
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def ensure_git_repo(repo_dir: Path) -> None:
    git_dir = repo_dir / ".git"
    if not git_dir.exists():
        sys.exit("이 디렉터리는 Git 저장소가 아닙니다.")


def ensure_origin_remote(repo_dir: Path) -> str:
    """Return the origin URL or exit with a clear message if missing."""
    code, stdout, stderr = run_cmd(["git", "remote", "get-url", "origin"], cwd=repo_dir)
    if code != 0:
        sys.exit(
            "원격 저장소 'origin' 이 설정되어 있지 않습니다.\n"
            "git remote add origin <repo-url> 명령으로 원격을 추가한 뒤 다시 실행하세요."
        )
    return stdout


def ensure_clean_worktree(repo_dir: Path) -> None:
    code, stdout, stderr = run_cmd(["git", "status", "--porcelain"], cwd=repo_dir)
    if code != 0:
        sys.exit(f"git status 실행 실패: {stderr}")
    if stdout:
        sys.exit("작업 디렉터리에 변경 사항이 있습니다. 커밋하거나 스태시한 후 다시 실행하세요.")


def detect_branch(repo_dir: Path) -> str:
    code, stdout, stderr = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir)
    if code != 0:
        sys.exit(f"현재 브랜치 확인 실패: {stderr}")
    return stdout


def fetch_and_pull(repo_dir: Path, branch: str) -> None:
    code, _, stderr = run_cmd(["git", "fetch", "--all", "--prune"], cwd=repo_dir)
    if code != 0:
        sys.exit(f"git fetch 실패: {stderr}")

    code, stdout, stderr = run_cmd(["git", "pull", "--ff-only", "origin", branch], cwd=repo_dir)
    if code != 0:
        sys.exit(f"git pull 실패: {stderr}")
    print(stdout)


def reinstall_dependencies(repo_dir: Path) -> None:
    requirements = repo_dir / "requirements.txt"
    if not requirements.exists():
        print("requirements.txt가 없어 의존성 설치를 건너뜁니다.")
        return

    print("pip install -r requirements.txt 실행 중...")
    code, stdout, stderr = run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo_dir)
    if stdout:
        print(stdout)
    if code != 0:
        sys.exit(f"의존성 설치 실패: {stderr}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GitHub 최신 변경사항 자동 패치 도구")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="requirements.txt 재설치를 건너뜁니다.",
    )
    return parser.parse_args()


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    args = parse_args()

    ensure_git_repo(repo_dir)
    ensure_origin_remote(repo_dir)
    ensure_clean_worktree(repo_dir)
    branch = detect_branch(repo_dir)

    print(f"현재 브랜치: {branch}")
    fetch_and_pull(repo_dir, branch)

    if args.skip_install:
        print("--skip-install 옵션으로 의존성 설치를 건너뜁니다.")
    else:
        reinstall_dependencies(repo_dir)

    print("업데이트 완료! streamlit run app.py 로 실행하세요.")


if __name__ == "__main__":
    main()
