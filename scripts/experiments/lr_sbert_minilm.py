"""Training pipeline for SBERT MiniLM embeddings-based spam classification.

This experiment uses SBERT with the all-MiniLM-L6-v2 model to generate semantic sentence embeddings.

WARNING: Test set results should ONLY be examined after selecting hyperparameters
based on dev set performance. Using test results for model selection leads to
overfitting and invalidates your evaluation.
"""

from sklearn.linear_model import LogisticRegression

from spam_ham_detector import (
    SBERT,
    CommentsDataset,
    evaluate_metrics,
    print_experiment_summary,
    save_model,
    save_predictions,
)


def main():
    """Load dataset, extract SBERT embeddings, and evaluate."""
    # ===== DATASET / PIPELINE =====
    dataset = CommentsDataset()
    embedder = SBERT(model_name='all-MiniLM-L6-v2')

    # ========== TRAIN SET ==========
    print('=' * 70 + '\n' + 'PROCESSING TRAIN SET' + '\n' + '=' * 70)
    X_train_emb = embedder.embed_documents(dataset.X_train)

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_emb, dataset.y_train)

    y_train_pred = classifier.predict(X_train_emb)
    y_train_proba = classifier.predict_proba(X_train_emb)[:, 1]
    train_metrics = evaluate_metrics(dataset.y_train, y_train_pred)

    # ========== DEV SET ==========
    print('=' * 70 + '\n' + 'PROCESSING DEV SET (for model selection)' + '\n' + '=' * 70)
    X_dev_emb = embedder.embed_documents(dataset.X_dev)

    y_dev_pred = classifier.predict(X_dev_emb)
    y_dev_proba = classifier.predict_proba(X_dev_emb)[:, 1]
    dev_metrics = evaluate_metrics(dataset.y_dev, y_dev_pred)

    # ========== FINAL SUMMARY ==========
    experiment = 'lr-sbert-minilm'
    save_model(experiment, classifier=classifier)
    save_predictions(dataset.X_train, dataset.y_train, y_train_pred, y_train_proba, experiment, 'train')
    save_predictions(dataset.X_dev, dataset.y_dev, y_dev_pred, y_dev_proba, experiment, 'dev')
    print_experiment_summary(train_metrics, dev_metrics, experiment_name=experiment)


if __name__ == '__main__':
    main()
