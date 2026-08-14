function ModeCard(props) {
  return (
    <div className="card">
      <h2>{props.title}</h2>

      <p>{props.description}</p>

      <button>{props.button}</button>
    </div>
  );
}

export default ModeCard;