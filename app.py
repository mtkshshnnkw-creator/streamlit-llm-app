import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st


SYSTEM_PROMPTS = {
    "データストラテジスト": (
        "あなたは事業データの活用に長けたデータストラテジストです。"
        "状況整理と次のアクション提案を日本語で具体的に行ってください。"
    ),
    "UXリサーチャー": (
        "あなたはユーザーインタビューに精通したUXリサーチャーです。"
        "ユーザー視点で課題を分析し、必要な調査手順を日本語で提案してください。"
    ),
}


def _load_api_key() -> str | None:
    """Prioritise Streamlit secrets and fall back to environment variables."""

    secret_key = st.secrets.get("OPENAI_API_KEY")
    if secret_key:
        return secret_key

    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def generate_response(user_text: str, persona: str) -> str:
    """Return the LLM answer based on the user input and selected persona."""

    system_prompt = SYSTEM_PROMPTS.get(persona, SYSTEM_PROMPTS["データストラテジスト"])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{user_input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"user_input": user_text})
def main() -> None:
    api_key = _load_api_key()
    if not api_key:
        st.error(
            "OPENAI_API_KEY が見つかりません。Streamlit Community Cloud では Secrets に必ず設定してください。"
        )
        st.stop()

    # ChatOpenAI は環境変数を参照するため、Secrets から取得した値を環境変数にも反映する
    os.environ.setdefault("OPENAI_API_KEY", api_key)

    st.title("LangChain × Streamlit デモ")
    st.write(
        "このアプリでは、ラジオボタンで専門家を切り替えながらテキスト入力を基に OpenAI GPT モデルに質問できます。"
    )
    st.caption("ラジオボタンで専門家を切り替え、入力内容を送信してください。")

    persona = st.radio("専門家を選択してください", list(SYSTEM_PROMPTS.keys()))

    with st.form(key="llm-form"):
        user_text = st.text_area("入力テキスト", placeholder="分析したい内容や相談ごとを入力してください。")
        submitted = st.form_submit_button("送信")

    if submitted:
        if not user_text.strip():
            st.warning("テキストを入力してください。")
            return

        with st.spinner("LLM に問い合わせ中..."):
            try:
                answer = generate_response(user_text=user_text, persona=persona)
            except Exception as exc:  # Surface API errors to the UI
                st.error(f"LLM からの応答取得に失敗しました: {exc}")
            else:
                st.subheader("LLMからの回答")
                st.write(answer)


if __name__ == "__main__":
    main()
