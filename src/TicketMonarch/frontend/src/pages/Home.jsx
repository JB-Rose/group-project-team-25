import { Link } from 'react-router-dom'
import './Home.css'
import chappellImg from '../assets/images/chappell.jpg'
import metallicaImg from '../assets/images/metallica.webp'
import gagaImg from '../assets/images/gaga.webp'
import linkinImg from '../assets/images/linkin.jpg'
import taylorImg from '../assets/images/taylor.png'

// real app fetch from api
const concerts = [
  {
    id: 1,
    name: 'Chappell Roan',
    date: 'Until Nov 16',
    eventName: 'Abono Banamex Plus Corona Capital 2025',
    location: 'México, CDMX, Mexico • Autodromo Hnos. Rodriguez',
    image: chappellImg,
    price: 100
  },
  {
    id: 2,
    name: 'Metallica',
    date: 'Until Nov 16',
    eventName: 'Abono Banamex Plus Corona Capital 2025',
    location: 'México, CDMX, Mexico • Autodromo Hnos. Rodriguez',
    image: metallicaImg,
    price: 250
  },
  {
    id: 3,
    name: 'Lady Gaga',
    date: 'Until Nov 16',
    eventName: 'Abono Banamex Plus Corona Capital 2025',
    location: 'México, CDMX, Mexico • Autodromo Hnos. Rodriguez',
    image: gagaImg,
    price: 200
  },
  {
    id: 4,
    name: 'Linkin Park',
    date: 'Until Nov 16',
    eventName: 'Abono Banamex Plus Corona Capital 2025',
    location: 'México, CDMX, Mexico • Autodromo Hnos. Rodriguez',
    image: linkinImg,
    price: 180
  },
  {
    id: 5,
    name: 'Taylor Swift',
    date: 'Until Nov 16',
    eventName: 'Abono Banamex Plus Corona Capital 2025',
    location: 'México, CDMX, Mexico • Autodromo Hnos. Rodriguez',
    image: taylorImg,
    price: 300
  }
]

function Home() {
  return (
    <div className="home-container">
      <header className="home-header">
        <div className="home-header-top">
          <div className="logo">
            <span className="logo-icon">🦋</span>
            <span className="logo-text">Ticket Monarch</span>
          </div>
          <div className="header-icons">
            <span className="icon">🛒</span>
            <span className="icon">☰</span>
          </div>
        </div>
        <div className="header-separator"></div>
      </header>

      <main className="home-main">
        <div className="concerts-list">
          {concerts.map(concert => (
            <div key={concert.id} className="concert-card">
              <img 
                src={concert.image} 
                alt={concert.name}
                className="concert-image"
              />
              <div className="concert-info">
                <h2 className="concert-name">{concert.name}</h2>
                <div className="concert-details">
                  <span className="concert-date">
                    {concert.date}
                    <span className="info-icon">ℹ️</span>
                  </span>
                  <p className="concert-event">{concert.eventName}</p>
                  <p className="concert-location">{concert.location}</p>
                </div>
              </div>
              <Link 
                to={`/seats/${concert.id}`}
                className="tickets-button"
              >
                Tickets →
              </Link>
            </div>
          ))}
        </div>
      </main>
    </div>
  )
}

export default Home