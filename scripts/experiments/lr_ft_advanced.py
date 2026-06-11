"""Training pipeline for FastText-based spam classification with advanced preprocessing.

This experiment uses advanced preprocessing (URL/email removal, lemmatization,
stopword filtering, POS tagging) to demonstrate the impact of linguistic feature selection with FastText embeddings.

WARNING: Test set results should ONLY be examined after selecting hyperparameters
based on dev set performance. Using test results for model selection leads to
overfitting and invalidates your evaluation.
"""

from sklearn.linear_model import LogisticRegression

from spam_ham_detector import (
    CommentsDataset,
    FastText,
    SpacyTokenizer,
    evaluate_metrics,
    print_experiment_summary,
    save_model,
    save_predictions,
)


def main():
    """Load dataset, extract embeddings with advanced preprocessing, and evaluate."""
    # ===== DATASET / PIPELINE =====
    dataset = CommentsDataset()

    pos_filter = ['N', 'V', 'J', 'R']  # Nouns, Verbs, Adjectives, Adverbs
    tokenizer = SpacyTokenizer(remove_noise=True, lemmatize=True, remove_stopwords=True, pos_filter=pos_filter)
    embedder = FastText()

    # ========== TRAIN SET ==========
    print('=' * 70 + '\n' + 'PROCESSING TRAIN SET' + '\n' + '=' * 70)
    X_train_tokens = tokenizer.process(dataset.X_train)
    X_train_emb = embedder.embed_documents(X_train_tokens)

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_emb, dataset.y_train)

    y_train_pred = classifier.predict(X_train_emb)
    y_train_proba = classifier.predict_proba(X_train_emb)[:, 1]
    train_metrics = evaluate_metrics(dataset.y_train, y_train_pred)

    # ========== DEV SET ==========
    print('=' * 70 + '\n' + 'PROCESSING DEV SET (for model selection)' + '\n' + '=' * 70)
    X_dev_tokens = tokenizer.process(dataset.X_dev)
    X_dev_emb = embedder.embed_documents(X_dev_tokens)

    y_dev_pred = classifier.predict(X_dev_emb)
    y_dev_proba = classifier.predict_proba(X_dev_emb)[:, 1]
    dev_metrics = evaluate_metrics(dataset.y_dev, y_dev_pred)

    # ========== FINAL SUMMARY ==========
    experiment = 'lr-ft-advanced'
    save_model(experiment, classifier=classifier)
    save_predictions(dataset.X_train, dataset.y_train, y_train_pred, y_train_proba, experiment, 'train')
    save_predictions(dataset.X_dev, dataset.y_dev, y_dev_pred, y_dev_proba, experiment, 'dev')
    print_experiment_summary(train_metrics, dev_metrics, experiment_name=experiment)


if __name__ == '__main__':
    main()
