// data.js — loads + parses MovieLens 100K (u.item, u.data)

let movies = [];           // [{id, title, year}]
let ratings = [];          // [{userId, movieId, rating}]
let numUsers = 0;
let numMovies = 0;

let userIdList = [];
let movieIdList = [];
let userIdToIndex = new Map();
let movieIdToIndex = new Map();

// point to u.item and u.data (local files in same repo folder)
const MOVIES_URL  = 'u.item';
const RATINGS_URL = 'u.data';

async function loadData() {
  const [movieResponse, ratingResponse] = await Promise.all([
    fetch(MOVIES_URL), fetch(RATINGS_URL)
  ]);
  if (!movieResponse.ok) throw new Error(`Movies fetch failed: ${movieResponse.status}`);
  if (!ratingResponse.ok) throw new Error(`Ratings fetch failed: ${ratingResponse.status}`);

  const [movieText, ratingText] = await Promise.all([
    movieResponse.text(), ratingResponse.text()
  ]);

  movies = parseItemData(movieText);
  ratings = parseRatingData(ratingText);

  userIdList = [...new Set(ratings.map(r => r.userId))].sort((a,b) => a-b);
  userIdList.forEach((id, i) => userIdToIndex.set(id, i));
  numUsers = userIdList.length;

  movieIdList = [...new Set(movies.map(m => m.id))].sort((a,b) => a-b);
  movieIdList.forEach((id, i) => movieIdToIndex.set(id, i));
  numMovies = movieIdList.length;

  console.log(`✅ Loaded ${numMovies} movies, ${ratings.length} ratings, ${numUsers} users`);
  return { movies, ratings, numUsers, numMovies };
}

function parseItemData(text) {
  const out = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const parts = line.split('|');
    if (parts.length < 2) continue;
    const id = parseInt(parts[0], 10);
    let title = parts[1].trim();
    let year = null;
    const m = title.match(/(.+)\s+\((\d{4})\)$/);
    if (m) { title = m[1].trim(); year = parseInt(m[2], 10); }
    out.push({ id, title, year });
  }
  return out;
}

function parseRatingData(text) {
  const out = [];
  for (const line of text.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    const [u, m, r] = s.split(/\s+/);
    if (u && m && r) {
      out.push({ userId: parseInt(u,10), movieId: parseInt(m,10), rating: parseFloat(r) });
    }
  }
  return out;
}
