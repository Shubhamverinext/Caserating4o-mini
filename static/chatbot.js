const form = document.querySelector('form');
const description = document.querySelector('#description');
const result = document.querySelector('#result');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    const userQuery = description.value.trim();
    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            description: userQuery
        })
    })
        .then((response) => response.json())
        .then((data) => {
            const caseType = data.case_type;
            const caseRating = data.case_rating;
            result.innerText = `Your query matches ${caseType} with a rating of ${caseRating}`;
        })
        .catch((error) => console.error(error));
});