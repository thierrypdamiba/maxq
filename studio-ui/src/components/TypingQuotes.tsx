"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Quote, X } from "lucide-react";

interface QuoteData {
    quote: string;
    author: string;
}

// Large fallback quote collection (100+ quotes)
function getFallbackQuotes(): QuoteData[] {
    return [
        { quote: "The only way to do great work is to love what you do.", author: "Steve Jobs" },
        { quote: "Innovation distinguishes between a leader and a follower.", author: "Steve Jobs" },
        { quote: "Stay hungry, stay foolish.", author: "Steve Jobs" },
        { quote: "The future belongs to those who believe in the beauty of their dreams.", author: "Eleanor Roosevelt" },
        { quote: "It always seems impossible until it's done.", author: "Nelson Mandela" },
        { quote: "The only impossible journey is the one you never begin.", author: "Tony Robbins" },
        { quote: "Success is not final, failure is not fatal: it is the courage to continue that counts.", author: "Winston Churchill" },
        { quote: "Believe you can and you're halfway there.", author: "Theodore Roosevelt" },
        { quote: "The best time to plant a tree was 20 years ago. The second best time is now.", author: "Chinese Proverb" },
        { quote: "Your time is limited, don't waste it living someone else's life.", author: "Steve Jobs" },
        { quote: "The way to get started is to quit talking and begin doing.", author: "Walt Disney" },
        { quote: "Don't watch the clock; do what it does. Keep going.", author: "Sam Levenson" },
        { quote: "The future depends on what you do today.", author: "Mahatma Gandhi" },
        { quote: "Everything you've ever wanted is on the other side of fear.", author: "George Addair" },
        { quote: "Success usually comes to those who are too busy to be looking for it.", author: "Henry David Thoreau" },
        { quote: "Don't be afraid to give up the good to go for the great.", author: "John D. Rockefeller" },
        { quote: "I find that the harder I work, the more luck I seem to have.", author: "Thomas Jefferson" },
        { quote: "Success is walking from failure to failure with no loss of enthusiasm.", author: "Winston Churchill" },
        { quote: "All progress takes place outside the comfort zone.", author: "Michael John Bobak" },
        { quote: "The only person you are destined to become is the person you decide to be.", author: "Ralph Waldo Emerson" },
        { quote: "Go confidently in the direction of your dreams. Live the life you have imagined.", author: "Henry David Thoreau" },
        { quote: "What lies behind us and what lies before us are tiny matters compared to what lies within us.", author: "Ralph Waldo Emerson" },
        { quote: "You must be the change you wish to see in the world.", author: "Mahatma Gandhi" },
        { quote: "The mind is everything. What you think you become.", author: "Buddha" },
        { quote: "The best revenge is massive success.", author: "Frank Sinatra" },
        { quote: "Life is 10% what happens to you and 90% how you react to it.", author: "Charles R. Swindoll" },
        { quote: "Do one thing every day that scares you.", author: "Eleanor Roosevelt" },
        { quote: "Definiteness of purpose is the starting point of all achievement.", author: "W. Clement Stone" },
        { quote: "Life isn't about finding yourself. Life is about creating yourself.", author: "George Bernard Shaw" },
        { quote: "Twenty years from now you will be more disappointed by the things that you didn't do than by the ones you did do.", author: "Mark Twain" },
        { quote: "The only limit to our realization of tomorrow will be our doubts of today.", author: "Franklin D. Roosevelt" },
        { quote: "It is during our darkest moments that we must focus to see the light.", author: "Aristotle" },
        { quote: "Whoever is happy will make others happy too.", author: "Anne Frank" },
        { quote: "Do not wait to strike till the iron is hot; but make it hot by striking.", author: "William Butler Yeats" },
        { quote: "Great things are done by a series of small things brought together.", author: "Vincent Van Gogh" },
        { quote: "What we think, we become.", author: "Buddha" },
        { quote: "Whether you think you can or you think you can't, you're right.", author: "Henry Ford" },
        { quote: "Strive not to be a success, but rather to be of value.", author: "Albert Einstein" },
        { quote: "Two roads diverged in a wood, and I—I took the one less traveled by, And that has made all the difference.", author: "Robert Frost" },
        { quote: "I attribute my success to this: I never gave or took any excuse.", author: "Florence Nightingale" },
        { quote: "You miss 100% of the shots you don't take.", author: "Wayne Gretzky" },
        { quote: "The most difficult thing is the decision to act, the rest is merely tenacity.", author: "Amelia Earhart" },
        { quote: "Every strike brings me closer to the next home run.", author: "Babe Ruth" },
        { quote: "Life is what happens to you while you're busy making other plans.", author: "John Lennon" },
        { quote: "We become what we think about.", author: "Earl Nightingale" },
        { quote: "The only way of discovering the limits of the possible is to venture a little way past them into the impossible.", author: "Arthur C. Clarke" },
        { quote: "An unexamined life is not worth living.", author: "Socrates" },
        { quote: "Eighty percent of success is showing up.", author: "Woody Allen" },
        { quote: "Winning isn't everything, but wanting to win is.", author: "Vince Lombardi" },
        { quote: "I am not a product of my circumstances. I am a product of my decisions.", author: "Stephen Covey" },
        { quote: "The most common way people give up their power is by thinking they don't have any.", author: "Alice Walker" },
        { quote: "You can't use up creativity. The more you use, the more you have.", author: "Maya Angelou" },
        { quote: "Dream big and dare to fail.", author: "Norman Vaughan" },
        { quote: "Everything has beauty, but not everyone can see.", author: "Confucius" },
        { quote: "How wonderful it is that nobody need wait a single moment before starting to improve the world.", author: "Anne Frank" },
        { quote: "Life is really simple, but we insist on making it complicated.", author: "Confucius" },
        { quote: "May you live every day of your life.", author: "Jonathan Swift" },
        { quote: "Life is a succession of lessons which must be lived to be understood.", author: "Helen Keller" },
        { quote: "Your time is limited, so don't waste it living someone else's life.", author: "Steve Jobs" },
        { quote: "If life were predictable it would cease to be life, and be without flavor.", author: "Eleanor Roosevelt" },
        { quote: "In the end, it's not the years in your life that count. It's the life in your years.", author: "Abraham Lincoln" },
        { quote: "Life is either a daring adventure or nothing at all.", author: "Helen Keller" },
        { quote: "Many of life's failures are people who did not realize how close they were to success when they gave up.", author: "Thomas Edison" },
        { quote: "The whole secret of a successful life is to find out what is one's destiny to do, and then do it.", author: "Henry Ford" },
        { quote: "In three words I can sum up everything I've learned about life: it goes on.", author: "Robert Frost" },
        { quote: "Love the life you live. Live the life you love.", author: "Bob Marley" },
        { quote: "Life is made of ever so many partings welded together.", author: "Charles Dickens" },
        { quote: "The purpose of our lives is to be happy.", author: "Dalai Lama" },
        { quote: "Get busy living or get busy dying.", author: "Stephen King" },
        { quote: "You only live once, but if you do it right, once is enough.", author: "Mae West" },
        { quote: "Not how long, but how well you have lived is the main thing.", author: "Seneca" },
        { quote: "The greatest glory in living lies not in never falling, but in rising every time we fall.", author: "Nelson Mandela" },
        { quote: "In order to write about life first you must live it.", author: "Ernest Hemingway" },
        { quote: "Life is not a problem to be solved, but a reality to be experienced.", author: "Soren Kierkegaard" },
        { quote: "The unexamined life is not worth living.", author: "Socrates" },
        { quote: "Good friends, good books, and a sleepy conscience: this is the ideal life.", author: "Mark Twain" },
        { quote: "Life is 10% what happens to me and 90% of how I react to it.", author: "Charles Swindoll" },
        { quote: "Keep smiling, because life is a beautiful thing and there's so much to smile about.", author: "Marilyn Monroe" },
        { quote: "Life is a long lesson in humility.", author: "James M. Barrie" },
        { quote: "In the depth of winter, I finally learned that within me there lay an invincible summer.", author: "Albert Camus" },
        { quote: "Life is never fair, and perhaps it is a good thing for most of us that it is not.", author: "Oscar Wilde" },
        { quote: "The only impossible journey is the one you never begin.", author: "Tony Robbins" },
        { quote: "In this life we cannot do great things. We can only do small things with great love.", author: "Mother Teresa" },
        { quote: "Change your thoughts and you change your world.", author: "Norman Vincent Peale" },
        { quote: "It is better to fail in originality than to succeed in imitation.", author: "Herman Melville" },
        { quote: "The road to success and the road to failure are almost exactly the same.", author: "Colin R. Davis" },
        { quote: "Success is not how high you have climbed, but how you make a positive difference to the world.", author: "Roy T. Bennett" },
        { quote: "Don't let yesterday take up too much of today.", author: "Will Rogers" },
        { quote: "You learn more from failure than from success. Don't let it stop you. Failure builds character.", author: "Unknown" },
        { quote: "If you are working on something that you really care about, you don't have to be pushed. The vision pulls you.", author: "Steve Jobs" },
        { quote: "People who are crazy enough to think they can change the world, are the ones who do.", author: "Rob Siltanen" },
        { quote: "We generate fears while we sit. We overcome them by action.", author: "Dr. Henry Link" },
        { quote: "Security is mostly a superstition. Life is either a daring adventure or nothing.", author: "Helen Keller" },
        { quote: "The only way to achieve the impossible is to believe it is possible.", author: "Charles Kingsleigh" },
        { quote: "We may encounter many defeats but we must not be defeated.", author: "Maya Angelou" },
        { quote: "Knowing is not enough; we must apply. Wishing is not enough; we must do.", author: "Johann Wolfgang Von Goethe" },
        { quote: "Imagine your life is perfect in every respect; what would it look like?", author: "Brian Tracy" },
        { quote: "We can easily forgive a child who is afraid of the dark; the real tragedy of life is when men are afraid of the light.", author: "Plato" },
        { quote: "Nothing in the world can take the place of Persistence.", author: "Calvin Coolidge" },
        { quote: "You are never too old to set another goal or to dream a new dream.", author: "C.S. Lewis" },
        { quote: "Try not to become a person of success, but rather try to become a person of value.", author: "Albert Einstein" },
        { quote: "A successful man is one who can lay a firm foundation with the bricks others have thrown at him.", author: "David Brinkley" },
    ].sort(() => Math.random() - 0.5); // Shuffle
}

export function TypingQuotes() {
    const [quotes, setQuotes] = useState<QuoteData[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [displayText, setDisplayText] = useState("");
    const [isTyping, setIsTyping] = useState(true);
    const [isDeleting, setIsDeleting] = useState(false);
    const [isEnabled, setIsEnabled] = useState(true);
    const [isLoading, setIsLoading] = useState(true);
    const [typosToMake, setTyposToMake] = useState(0);
    const [typosMade, setTyposMade] = useState(0);
    const [isFixingTypo, setIsFixingTypo] = useState(false);
    const [correctTextLength, setCorrectTextLength] = useState(0);
    const containerRef = useRef<HTMLDivElement | null>(null);

    // Auto-scroll to keep cursor visible
    useEffect(() => {
        if (containerRef.current) {
            const container = containerRef.current;
            const textWidth = container.scrollWidth;
            const containerWidth = container.clientWidth;

            // Only scroll if content is wider than container
            if (textWidth > containerWidth) {
                // Scroll to show the end (cursor) but keep as much of the beginning visible as possible
                container.scrollLeft = textWidth - containerWidth;
            } else {
                // If it fits, stay at the beginning
                container.scrollLeft = 0;
            }
        }
    }, [displayText]);

    // Fetch quotes from API-Ninjas
    useEffect(() => {
        // Start with fallback quotes immediately for instant display
        setQuotes(getFallbackQuotes());
        setIsLoading(false);

        // Optionally try to fetch more quotes from API in the background
        const fetchQuotes = async () => {
            // Only fetch if API key is available
            if (!process.env.NEXT_PUBLIC_API_NINJAS_KEY) {
                return;
            }

            try {
                const categories = ["happiness", "success", "inspirational", "life", "wisdom", "knowledge", "future"];
                const allQuotes: QuoteData[] = [];

                for (const category of categories) {
                    try {
                        const response = await fetch(`https://api.api-ninjas.com/v1/quotes?category=${category}&limit=30`, {
                            headers: {
                                'X-Api-Key': process.env.NEXT_PUBLIC_API_NINJAS_KEY || '',
                            },
                        });

                        if (response.ok) {
                            const data = await response.json();
                            allQuotes.push(...data);
                        }
                    } catch (err) {
                        console.error(`Failed to fetch ${category} quotes:`, err);
                    }
                }

                // If we got quotes from API, merge with fallback and update
                if (allQuotes.length > 0) {
                    const uniqueQuotes = [...getFallbackQuotes(), ...allQuotes].filter(
                        (quote, index, self) =>
                            index === self.findIndex((q) => q.quote === quote.quote)
                    );
                    const shuffled = uniqueQuotes.sort(() => Math.random() - 0.5);
                    setQuotes(shuffled);
                }
            } catch (error) {
                console.error("Failed to fetch additional quotes:", error);
            }
        };

        fetchQuotes();
    }, []);

    // Initialize typos for new quote
    useEffect(() => {
        if (!isDeleting && displayText.length === 0 && quotes.length > 0) {
            // Decide how many typos to make for this quote
            const rand = Math.random();
            let typos = 0;
            if (rand < 0.50) typos = 1;      // 50% chance: 1 typo
            else if (rand < 0.75) typos = 0; // 25% chance: 0 typos
            else if (rand < 0.90) typos = 2; // 15% chance: 2 typos
            else typos = 3;                   // 10% chance: 3 typos

            setTyposToMake(typos);
            setTyposMade(0);
        }
    }, [isDeleting, displayText, quotes.length]);

    // Typing animation effect with controlled typos
    useEffect(() => {
        if (!isEnabled || quotes.length === 0 || isLoading) return;

        const currentQuote = quotes[currentIndex];
        if (!currentQuote) return;

        const fullText = `"${currentQuote.quote}" — ${currentQuote.author}`;

        if (isTyping && !isDeleting && !isFixingTypo) {
            if (displayText.length < fullText.length) {
                // Check if we should make a typo
                const shouldMakeTypo = typosMade < typosToMake &&
                                      displayText.length > 15 &&
                                      displayText.length < fullText.length - 20 &&
                                      Math.random() < 0.03; // Small chance each keystroke

                if (shouldMakeTypo) {
                    // Make a typo - type a wrong character
                    const wrongChars = 'abcdefghijklmnopqrstuvwxyz';
                    const wrongChar = wrongChars[Math.floor(Math.random() * wrongChars.length)];
                    const timeout = setTimeout(() => {
                        setDisplayText(displayText + wrongChar);
                        setCorrectTextLength(displayText.length);
                        setTyposMade(prev => prev + 1);
                        setIsFixingTypo(true);
                    }, 40 + Math.random() * 40);
                    return () => clearTimeout(timeout);
                } else {
                    // Type correct character
                    const timeout = setTimeout(() => {
                        setDisplayText(fullText.slice(0, displayText.length + 1));
                    }, 40 + Math.random() * 40);
                    return () => clearTimeout(timeout);
                }
            } else {
                // Finished typing - pause before deleting
                const timeout = setTimeout(() => {
                    setIsDeleting(true);
                }, 2500);
                return () => clearTimeout(timeout);
            }
        } else if (isFixingTypo) {
            // Delete the typo character
            if (displayText.length > correctTextLength) {
                const timeout = setTimeout(() => {
                    setDisplayText(displayText.slice(0, -1));
                }, 30); // Fast backspace
                return () => clearTimeout(timeout);
            } else {
                // Done fixing typo, resume typing
                setIsFixingTypo(false);
            }
        } else if (isDeleting) {
            if (displayText.length > 0) {
                const timeout = setTimeout(() => {
                    setDisplayText(displayText.slice(0, -1));
                }, 25); // Slower, more visible deletion
                return () => clearTimeout(timeout);
            } else {
                // Move to next quote
                setIsDeleting(false);
                setCurrentIndex((prev) => (prev + 1) % quotes.length);
            }
        }
    }, [displayText, isTyping, isDeleting, isFixingTypo, isEnabled, quotes, currentIndex, isLoading, typosToMake, typosMade, correctTextLength]);

    if (!isEnabled) {
        return (
            <button
                onClick={() => setIsEnabled(true)}
                className="text-foreground-muted/30 hover:text-foreground-muted transition-colors"
                title="Enable quotes"
            >
                <Quote className="w-4 h-4" />
            </button>
        );
    }

    return (
        <div className="flex items-center gap-2">
            <div
                ref={containerRef}
                className="bg-black/40 backdrop-blur-sm border border-white/5 rounded-md px-3 py-1.5 max-w-3xl overflow-x-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
            >
                <p className="text-xs text-foreground-muted/70 font-mono italic whitespace-nowrap">
                    {displayText}
                    <span className="animate-pulse">|</span>
                </p>
            </div>
            <button
                onClick={() => setIsEnabled(false)}
                className="text-foreground-muted/30 hover:text-foreground-muted transition-colors flex-shrink-0"
                title="Disable quotes"
            >
                <X className="w-3 h-3" />
            </button>
        </div>
    );
}
