# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード（キャッシュを利用）
# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None
pipe = llm.load_model()

# --- Streamlit アプリケーション ---
st.title("🤖 Gemma 2 Chatbot with Feedback")
st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "チャット" # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page), # 現在のページを選択状態にする
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector) # 選択変更時に状態を更新
)

# モデルパラメータセクションを追加
st.sidebar.markdown("---")
st.sidebar.title("モデルパラメータ")

# 初期値の設定（セッションステートがなければデフォルト値を使用）
if 'max_length' not in st.session_state:
    st.session_state.max_length = 512
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'top_p' not in st.session_state:
    st.session_state.top_p = 0.9

# スライダーの追加
max_length = st.sidebar.slider(
    "回答の最大長", 
    min_value=64, 
    max_value=2048, 
    value=st.session_state.max_length, 
    step=64,
    help="生成される回答の最大トークン数を指定します"
)
st.session_state.max_length = max_length

temperature = st.sidebar.slider(
    "温度 (Temperature)", 
    min_value=0.0, 
    max_value=2.0, 
    value=st.session_state.temperature, 
    step=0.1,
    help="値が高いほど創造的な回答になり、低いほど決定的な回答になります"
)
st.session_state.temperature = temperature

top_p = st.sidebar.slider(
    "多様性 (Top P)", 
    min_value=0.0, 
    max_value=1.0, 
    value=st.session_state.top_p, 
    step=0.05,
    help="次のトークンを選択する際の確率閾値を設定します"
)
st.session_state.top_p = top_p



# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")